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
from typing import Any

import yaml

# Hide mypy error `Module "labjack" has no attribute "ljm"`.
from labjack import ljm  # type: ignore
from lsst.ts import salobj

from .base_labjack_data_client import BaseLabJackDataClient
from .topic_handler import TopicHandler

# Time limit for communicating with the LabJack (seconds).
# This includes writing a command and reading the response
# and reading telemetry (seconds).
READ_TIMEOUT = 5


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
        # List of LabJack channel names to read.
        self.channel_names: Sequence[str] = []

        # dict of (topic_attr_name, sensor_name): TopicHandler.
        self.topic_handlers: dict[tuple[str, str], TopicHandler] = dict()

        # An event that unit tests can use to wait for data to be written.
        # A test can clear the event, then wait for it to be set.
        self.wrote_event = asyncio.Event()

        # Mock data for simulation mode; if None then use whatever values
        # LabJack ljm returns (which is not documented).
        # If specified, it must include one value per channel,
        # in the same order as self.channel_names.
        self.mock_raw_data: None | Sequence[float] = None

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
  topics:
    description: >-
      Information about the ESS SAL topics this client writes. Note that this
      data client only supports array-valued topics, such as tel_temperature.
    type: array
    minItems: 1
    items:
      type: object
      properties:
        topic_name:
          description: SAL topic attribute name, e.g. tel_temperature.
          type: string
        sensor_name:
          description: Value for the sensor_name field of the topic.
          type: string
        field_name:
          description: Name of array-valued SAL topic field.
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
        offset:
          description: SAL value = (LabJack value - offset) * scale
          type: number
          default: 0
        scale:
          description: SAL value = (LabJack value - offset) * scale
          type: number
          default: 1
      required:
        - topic_name
        - sensor_name
        - field_name
        - location
        - channel_names
        - offset
        - scale
      additionalProperties: false
required:
  - device_type
  - connection_type
  - identifier
  - poll_interval
  - topics
additionalProperties: false
"""
        )

    def configure(self) -> None:
        """Store device configurations.

        Also initialize all output arrays to NaNs.

        This provides easy access when processing telemetry.
        """
        # set of all LabJack channel names to read.
        channel_names: set[str] = set()
        for topic_info_dict in self.config.topics:
            topic_info = types.SimpleNamespace(**topic_info_dict)
            topic_name = topic_info.topic_name
            topic_key = (topic_name, topic_info.sensor_name)

            if topic_key in self.topic_handlers:
                raise RuntimeError(
                    f"topic_name={topic_name} with sensor_name={topic_info.sensor_name} "
                    "listed more than once"
                )
            topic = getattr(self.topics, topic_name, None)
            if topic is None:
                raise RuntimeError(f"topic {topic_name} not found")
            topic_handler = TopicHandler(
                topic=topic,
                sensor_name=topic_info.sensor_name,
                field_name=topic_info.field_name,
                location=topic_info.location,
                channel_names=topic_info.channel_names,
                offset=topic_info.offset,
                scale=topic_info.scale,
            )
            self.topic_handlers[topic_key] = topic_handler
            new_channel_names = set(topic_handler.channel_dict.values())
            overlap_channel_names = new_channel_names & channel_names
            if overlap_channel_names:
                raise ValueError(
                    f"Channel names {sorted(overlap_channel_names)} appear more than once."
                )
            channel_names |= new_channel_names
        self.channel_names = tuple(sorted(channel_names))

    async def run(self) -> None:
        """Read and process data from the LabJack."""
        while True:
            data_dict = await self.run_in_thread(
                func=self._blocking_read, timeout=READ_TIMEOUT
            )
            for topic_handler in self.topic_handlers.values():
                await topic_handler.put_data(data_dict)
            # Support unit testing with a future the test can reset.
            self.wrote_event.set()
            await asyncio.sleep(self.config.poll_interval)

    def _blocking_connect(self) -> None:
        """Connect and then read the specified channels.

        This makes sure that the configured channels can be read.
        """
        # Read each input channel, to make sure the configuration is valid.
        super()._blocking_connect()
        self._blocking_read()

    def _blocking_read(self) -> dict[str, float]:
        """Read data from the LabJack. This can block.

        Call in a thread to avoid blocking the event loop.

        Returns
        -------
        data : `dict` [`str`, `float`]
            The read data as a dict of channel_name: value.
        """
        if self.handle is None:
            raise RuntimeError("Not connected")

        num_frames = len(self.channel_names)
        values = ljm.eReadNames(self.handle, num_frames, self.channel_names)
        if self.simulation_mode != 0 and self.mock_raw_data is not None:
            assert len(self.mock_raw_data) == len(values)
            values = self.mock_raw_data
        if len(values) != len(self.channel_names):
            raise RuntimeError(
                f"len(channel_names)={self.channel_names} != len(values)={values}"
            )
        return {name: value for name, value in zip(self.channel_names, values)}
