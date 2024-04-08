# This file is part of ts_ess_labjack.
#
# Developed for the Vera C. Rubin Observatory Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["LabJackDataClient"]

import asyncio
import logging
import types
from collections.abc import Sequence
from typing import Any, Type

import yaml

# Hide mypy error `Module "labjack" has no attribute "ljm"`.
from labjack import ljm  # type: ignore
from lsst.ts import salobj, utils
from lsst.ts.ess import common

from .base_labjack_data_client import BaseLabJackDataClient

# Time limit for communicating with the LabJack (seconds).
# This includes writing a command and reading the response
# and reading telemetry (seconds).
READ_TIMEOUT = 5

# Sleep time before trying to reconnect (seconds).
RECONNECT_WAIT = 60

# Maximum number of allowed timeouts before throwing an error.
MAX_TIMEOUTS = 5


class LabJackDataClient(BaseLabJackDataClient):
    """Get environmental data from a LabJack T7 or similar.

    Parameters
    ----------
    name : str
    config : types.SimpleNamespace
        The configuration, after validation by the schema returned
        by `get_config_schema` and conversion to a types.SimpleNamespace.
    topics : `salobj.Controller`
        The telemetry topics this model can write, as a struct with attributes
        such as ``tel_temperature``.
    log : `logging.Logger`
        Logger.
    simulation_mode : `int`, optional
        Simulation mode; 0 for normal operation.
    """

    def __init__(
        self,
        config: types.SimpleNamespace,
        topics: salobj.Controller | types.SimpleNamespace,
        log: logging.Logger,
        simulation_mode: int = 0,
    ) -> None:
        super().__init__(
            config=config, topics=topics, log=log, simulation_mode=simulation_mode
        )
        # The telemetry processor.
        self.processor: common.processor.BaseProcessor | None = None
        # List of LabJack channel names to read.
        self.channel_names: Sequence[str] = []
        # List of offsets.
        self.offsets: Sequence[str] = []
        # List of scales.
        self.scales: Sequence[str] = []

        # Dict of SensorType: BaseProcessor type.
        self.telemetry_processor_dict: dict[
            str, Type[common.processor.BaseProcessor]
        ] = {
            "AirTurbulenceProcessor": common.processor.AirTurbulenceProcessor,
            "AuxTelCameraCoolantPressureProcessor": common.processor.AuxTelCameraCoolantPressureProcessor,
        }

        # An event that unit tests can use to wait for data to be written.
        # A test can clear the event, then wait for it to be set.
        self.wrote_event = asyncio.Event()

        # Mock data for simulation mode; if None then use whatever values
        # LabJack ljm returns (which is not documented).
        # If specified, it must include one value per channel,
        # in the same order as self.channel_names.
        self.mock_raw_data: Sequence[float] | None = None

        self.configure()

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        return yaml.safe_load(
            """
$schema: http://json-schema.org/draft-07/schema#
description: Schema for LabJackDataClient
type: object
properties:
  device_type:
    description: LabJack model
    type: string
    default: T7
  connection_type:
    description: Connection type
    type: string
    default: TCP
  identifier:
    description: >-
        LabJack indentifier:
        * A host name or IP address if connection_type = TCP or WIFI
        * A serial number if connection_type = USB
        * For testing in an environment with only one LabJack you may use ANY
    type: string
  poll_interval:
    description: Polling interval (seconds)
    type: number
    default: 1
  processor:
    description: The telemetry processor.
    type: string
    enum:
      - AirTurbulenceProcessor
      - AuxTelCameraCoolantPressureProcessor
  sensor_name:
    description: Value for the sensor_name field of the topic.
    type: string
  location:
    description: >-
      Location of sensors. A comma-separated list,
      with one item per non-null channel_name.
    type: string
  channel_names:
    description: >-
      LabJack channel names, in order of the field array.
      Specify empty strings for skipped channels.
      Here is an example that sets temperature indices 0, 2, and 3
      (skipping index 1): [AIN05, "", AIN07, AIN06]
    type: array
    minItems: 1
    items:
      type: string
  offsets:
    description: >-
      Array of offsets, one per channel.
      SAL value = (LabJack value - offset) * scale
    type: array
    minItems: 1
    items:
      type: number
    default: 0
  scales:
    description: >-
      Array of scales, one per channel.
      SAL value = (LabJack value - offset) * scale
    type: array
    minItems: 1
    items:
      type: number
    default: 1
  num_samples:
    description: >-
      Number of samples per telemetry sample. Only relevant for
      certain kinds of data, such as wind speed and direction.
      Ignored for other kinds of data.
    type: integer
    minimum: 2
required:
  - device_type
  - connection_type
  - identifier
  - poll_interval
  - processor
  - sensor_name
  - location
  - channel_names
  - offsets
  - scales
"""
        )

    def configure(self) -> None:
        """Store device configurations.

        Also initialize all output arrays to NaNs.

        This provides easy access when processing telemetry.
        """
        device_config = common.DeviceConfig(
            name=self.config.sensor_name,
            dev_type=None,
            dev_id="",
            sens_type=None,
            baud_rate=0,
            location=self.config.location,
        )
        if hasattr(self.config, "num_samples"):
            device_config.num_samples = self.config.num_samples
        processor_type = self.telemetry_processor_dict[self.config.processor]
        self.processor = processor_type(device_config, self.topics, self.log)
        self.channel_names = self.config.channel_names
        self.offsets = self.config.offsets
        self.scales = self.config.scales
        if len(self.offsets) != len(self.channel_names):
            raise ValueError(
                f"The number of offsets {len(self.offsets)} doesn't correspond "
                f"to the number of channels {len(self.channel_names)}."
            )
        if len(self.scales) != len(self.channel_names):
            raise ValueError(
                f"The number of scales {len(self.scales)} doesn't correspond "
                f"to the number of channels {len(self.channel_names)}."
            )

    async def read_data(self) -> None:
        """Read and process data from the LabJack."""
        try:
            telemetry = await self.run_in_thread(
                func=self._blocking_read, timeout=READ_TIMEOUT
            )
            assert self.processor is not None
            await self.processor.process_telemetry(
                timestamp=utils.current_tai(),
                response_code=0,
                sensor_data=telemetry,
            )
            # Support unit testing with a future the test can reset.
            self.wrote_event.set()
            await asyncio.sleep(self.config.poll_interval)
        except asyncio.TimeoutError:
            self.log.warning(f"Timeout. Trying again in {RECONNECT_WAIT} seconds.")
            await asyncio.sleep(RECONNECT_WAIT)
            raise

    def _blocking_connect(self) -> None:
        """Connect and then read the specified channels.

        This makes sure that the configured channels can be read.
        """
        # Read each input channel, to make sure the configuration is valid.
        super()._blocking_connect()
        self._blocking_read()

    def _blocking_read(self) -> list[float]:
        """Read telemetry from the LabJack. This can block.

        Call in a thread to avoid blocking the event loop.

        Returns
        -------
        `list` [`float`]
            The read telemetry as a list of values.
        """
        if self.handle is None:
            raise RuntimeError("Not connected")

        num_frames = len(self.channel_names)
        values = ljm.eReadNames(self.handle, num_frames, self.channel_names)
        if self.simulation_mode != 0 and self.mock_raw_data is not None:
            assert len(self.mock_raw_data) == len(values)
            values = self.mock_raw_data
        else:
            self.log.debug(
                "read values %s from channels %s", values, self.channel_names
            )
        if len(values) != len(self.channel_names):
            raise RuntimeError(
                f"len(channel_names)={self.channel_names} != len(values)={values}"
            )
        converted_values = []
        # Apply the corresponding offset and scale to each value.
        for i in range(len(values)):
            converted_values.append((values[i] - self.offsets[i]) * self.scales[i])
        return converted_values
