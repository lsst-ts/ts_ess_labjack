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

__all__ = ["LabJackAccelerometerDataClient"]

import asyncio
import logging
import math
import types
from typing import Any

# Hide my error `Module "labjack" has no attribute "ljm"`
from labjack import ljm  # type: ignore
import numpy as np
import yaml

from lsst.ts import salobj
from lsst.ts import utils
from .base_labjack_data_client import BaseLabJackDataClient

# Time limit for communicating with the LabJack (seconds)
COMMUNICATION_TIMEOUT = 5

# Maximum sampling frequency (Hz) that the simulator allows
MAX_MOCK_SAMPLE_FREQUENCY = 1000

# Smallest allowed max_frequency / config.min_frequency
MIN_FREQUENCY_RATIO = 2


class LabJackAccelerometerDataClient(BaseLabJackDataClient):
    """Read 3-axis accelerometer data from a LabJack T7 or similar,
    and report using the accelerometerPSD topic.

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

    Notes
    -----
    In simulation mode the mock LabJack returns unspecified values,
    and those values may change in future versions of the LabJack software.
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

        self.topic = topics.tel_accelerometerPSD
        self.array_len = len(self.topic.DataType().accelerationPSDX)

        # Read three channels; X, Y, Z
        self.num_channels = 3  # x, y, z

        if config.min_frequency * MIN_FREQUENCY_RATIO > config.max_frequency:
            raise ValueError(
                f"{self.config.min_frequency=} must be < "
                f"{config.max_frequency=} / {MIN_FREQUENCY_RATIO=} "
                f"= {config.max_frequency / MIN_FREQUENCY_RATIO}"
            )

        # Interval between samples (seconds);
        # Set by _blocking_start_data_stream
        # since the LabJack may offer a smaller value than requested.
        self.sampling_interval: None | float = None

        # Number of samples (per channel) to measure PSD
        num_frequencies_from_0 = round(
            config.num_frequencies
            * config.max_frequency
            / (config.max_frequency - config.min_frequency)
        )
        self.num_samples = num_frequencies_from_0 * 2 - 2

        # Starting index of PSD, to get the subset of frequencies
        # that we want to report. Set in _blocking_start_data_stream.
        self.psd_start_index: None | int = None

        # Frequencies of the PSD, starting from 0.
        # Set in _blocking_start_data_stream.
        self.psd_frequencies: None | np.ndarray = None

        assert len(self.config.analog_inputs) == self.num_channels
        self.topic.set(
            sensorName=config.sensor_name,
            location=config.location,
            numDataPoints=0,
            **{
                f"accelerationPSD{axis}": [math.nan] * self.array_len
                for axis in ("X", "Y", "Z")
            },
        )
        self.modbus_addresses: list[int] = [ai * 2 for ai in self.config.analog_inputs]
        self.loop = asyncio.get_running_loop()
        self.mock_stream_task = utils.make_done_future()

        # In simulation mode controls whether the client generates
        # random mock_raw_1d_data (producing garbage PSDs).
        # By default it is True, so that some output is produced.
        # Set it to False if you would rather generate the raw data yourself,
        # in which case the topic is not written until
        # you set self.mock_raw_1d_data.
        self.make_random_mock_raw_1d_data = True
        # Unit tests may set this to a list of floats
        # with length self.num_samples containing:
        # [v0_chan0, v0_chan1, v0_chan2, v1_chan0, v1_chan1, v1_chan2,
        # ..., vn_chan0, vn_chan1, vn_chan2]
        # In simulation mode this will be used used to compute the PSDs.
        # If None and if make_random_mock_raw_1d_data is true
        # then it will be set to random data when first needed.
        self.mock_raw_1d_data: None | list[float] = None

        # An event that unit tests can use to wait for data to be written.
        # A test can clear the event, then wait for it to be set.
        self.wrote_event = asyncio.Event()

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        return yaml.safe_load(
            f"""
$schema: http://json-schema.org/draft-07/schema#
description: Schema for LabJackAccelerometerDataClient
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
  sensor_name:
    description: Value for the sensor_name field of the topic.
    type: string
  location:
    description: Location of sensor.
    type: string
  min_frequency:
    description: >-
        Approximate minimum frequency to report in the PSD (Hz).
        A value larger than 0 may be used to increase the resolution
        of the reported PSD.
        Must be ≤ max_frequency / {MIN_FREQUENCY_RATIO=}.
    type: number
    default: 0
    minimum: 0
  max_frequency:
    description: >-
        Maximum frequency to report in the PSD (Hz).
        If larger than the LabJack can handle then it is reduced.
    type: number
    minimum: 0
  num_frequencies:
    description: >-
        Number of frequencies to report in the PSD.
        Must be less than or equal to the length of the PSD array fields.
    type: integer
    minValue: 10
    default: 200
  analog_inputs:
    description: >-
        Analog inputs read for x, y, and z data.
        0 for AIN0, 1 for AIN1, 2 or AIN2, etc.
    type: array
    minItems: 3
    maxItems: 3
    items:
        type: integer
        minimum: 0
  scale:
    description: Accelerometer scale (m/s2 per Volt).
    type: number
required:
  - device_type
  - connection_type
  - identifier
  - sensor_name
  - location
  - min_frequency
  - max_frequency
  - num_frequencies
  - analog_inputs
  - scale
additionalProperties: false
"""
        )

    def descr(self) -> str:
        return f"identifier={self.config.identifier}"

    async def run(self) -> None:
        """Read and process data from the LabJack."""
        await self.start_data_stream()
        await asyncio.Future()

    async def disconnect(self) -> None:
        self.mock_stream_task.cancel()
        if self.handle is not None:
            await self.stop_data_stream()
        await super().disconnect()

    async def start_data_stream(self) -> None:
        """Start the data stream from the LabJack."""

        return await self.run_in_thread(
            func=self._blocking_start_data_stream, timeout=COMMUNICATION_TIMEOUT
        )

    def start_mock_stream(self) -> None:
        """Start a mock stream, since ljm demo mode does not stream."""
        if self.simulation_mode == 0:
            raise ValueError("start_mock_stream can only be called in simulation mode")

        self.mock_stream_task.cancel()
        self.mock_stream_task = asyncio.create_task(self._mock_stream())

    async def _mock_stream(self) -> None:
        """Pretend to stream data.

        Stream self.mock_raw_1d_data.
        If that is None then set it to a random array.
        """
        self.log.info("mock_stream begins")
        if self.sampling_interval is None:
            raise RuntimeError("Streaming not started")
        try:
            sleep_interval = self.sampling_interval * self.num_samples
            if self.mock_raw_1d_data is None and self.make_random_mock_raw_1d_data:
                # Generate random mock data
                # using half the available scale of -10 to 10 volts
                self.mock_raw_1d_data = list(
                    np.random.random(self.num_samples * self.num_channels) * 10 - 5
                )
            while True:
                await asyncio.sleep(sleep_interval)
                if self.mock_raw_1d_data is not None:
                    scaled_data = self.scaled_data_from_raw(self.mock_raw_1d_data)
                await self.callback(scaled_data, (0, 0))
        except asyncio.CancelledError:
            self.log.info("mock_stream ends")
        except Exception:
            self.log.exception("mock_stream failed")
            raise

    def _blocking_start_data_stream(self) -> None:
        """Start streaming data from the LabJack

        Call in a thread to avoid blocking the event loop.
        """
        self.mock_stream_task.cancel()

        desired_sampling_frequency = 2 * self.config.max_frequency

        # LabJack ljm demo mode does not support streaming,
        # so use mock streaming
        if self.simulation_mode == 0:
            actual_sampling_frequency = ljm.eStreamStart(
                self.handle,
                self.num_samples,
                len(self.modbus_addresses),
                self.modbus_addresses,
                desired_sampling_frequency,
            )
            self.log.info(
                f"{desired_sampling_frequency=}, {actual_sampling_frequency=}"
            )
            # Warn if the LabJack cannot gather data as quickly as requested.
            # Allow a bit of margin for roundoff error (the log statement
            # above may help determine a good value for this margin)
            if actual_sampling_frequency < desired_sampling_frequency * 0.99:
                actual_max_frequency = actual_sampling_frequency / 2
                self.log.warning(
                    "LabJack cannot gather data that quickly; "
                    f"{self.config.max_frequency=} reduced to {actual_max_frequency}"
                )
            self.sampling_interval = 1 / actual_sampling_frequency
            ljm.setStreamCallback(self.handle, self.blocking_data_stream_callback)
        else:
            actual_sampling_frequency = min(
                desired_sampling_frequency, MAX_MOCK_SAMPLE_FREQUENCY
            )
            if desired_sampling_frequency > MAX_MOCK_SAMPLE_FREQUENCY:
                actual_max_frequency = actual_sampling_frequency / 2
                self.log.warning(
                    "Mock LabJack cannot gather data that quickly; "
                    f"{self.config.max_frequency=} reduced to {actual_max_frequency}"
                )
            self.sampling_interval = 1 / actual_sampling_frequency
            self.loop.call_soon_threadsafe(self.start_mock_stream)

        # Compute self.psd_frequencies and self.psd_start_index
        self.psd_frequencies = np.fft.rfftfreq(self.num_samples, self.sampling_interval)
        assert self.psd_frequencies is not None  # make mypy happy

        num_frequencies_measured = len(self.psd_frequencies)
        if num_frequencies_measured < self.config.num_frequencies:
            self.log.warning(
                f"Bug! {self.config.num_frequencies=} too large; reduced to {num_frequencies_measured}"
            )
            self.config.num_frequencies = num_frequencies_measured
        self.psd_start_index = num_frequencies_measured - self.config.num_frequencies

        self.log.info(
            f"actual min_frequency={self.psd_frequencies[self.psd_start_index]:0.2f}, "
            f"max_frequency={self.psd_frequencies[-1]:0.2f}"
        )

    def _blocking_stop_data_stream(self) -> None:
        """Stop streaming data from the LabJack

        Call in a thread to avoid blocking the event loop.
        """
        # LabJack ljm demo mode does not support streaming,
        # but this call seems to work anyway
        ljm.eStreamStop(self.handle)

    def blocking_data_stream_callback(self, handle: int) -> None:
        """Called in a thread when a full set of stream data is available."""
        (
            raw_1d_data,
            backlog1,
            backlog2,
        ) = ljm.eStreamRead(self.handle)
        scaled_data = self.scaled_data_from_raw(raw_1d_data)
        self.loop.call_soon_threadsafe(self.callback, scaled_data, (backlog1, backlog2))

    def scaled_data_from_raw(self, raw_1d_data: list[float]) -> np.ndarray:
        """Convert a list of 1-D raw readings to 2-d scaled data.

        Parameters
        ----------
        raw_1d_data : list[float]
            Raw voltage data read from the LabJack in the form:
            [v0_chan0, v0_chan1, v0_chan2, v1_chan0, v1_chan1, v1_chan2,
            ..., vn_chan0, vn_chan1, vn_chan2]

        Returns
        -------
        scaled_data : np.ndarray
            Raw data multiplied by self.config.scale
            and reshaped to size (num_channels, voltages)
        """
        npoints = len(raw_1d_data) // self.num_channels
        return (
            np.reshape(raw_1d_data, newshape=(self.num_channels, npoints), order="F")
            * self.config.scale
        )

    async def callback(
        self, scaled_data: np.ndarray, backlogs: tuple[int, int]
    ) -> None:
        """Process one set of data.

        Parameters
        ----------
        scaled_data : `np.ndarray`
            x, y, z acceleration data of shape (self.num_channels, n)
            after scaling by self.config.scale
        backlogs : `tuple`
            Two measures of the number of backlogged messages.
            Both values should be nearly zero if the data client
            is keeping up with the LabJack.

        Raises
        ------
        RuntimeError
            If streaming has not yet begun (because self.sampling_interval
            is None until streaming begins).
        """
        if self.psd_start_index is None or self.psd_frequencies is None:
            raise RuntimeError("Sampling has not yet been configured")

        if scaled_data.shape != (self.num_channels, self.num_samples):
            self.log.error(
                f"Bug: {scaled_data.shape=} != ({self.num_channels}, {self.num_samples}); "
                "callback ignoring this data"
            )
            return

        try:
            psd = np.abs(np.fft.rfft(np.array(scaled_data))) ** 2
            psd_kwargs = {
                f"accelerationPSD{axis}": psd[i, self.psd_start_index :]
                for i, axis in enumerate(("X", "Y", "Z"))
            }
            await self.topic.set_write(
                interval=self.sampling_interval,
                minPSDFrequency=self.psd_frequencies[self.psd_start_index],
                maxPSDFrequency=self.psd_frequencies[-1],
                numDataPoints=self.config.num_frequencies,
                **psd_kwargs,
            )
            self.wrote_event.set()
        except Exception:
            self.log.exception("callback failed")
            raise

    async def stop_data_stream(self) -> None:
        """Stop the data stream from the LabJack."""
        self.mock_stream_task.cancel()
        await self.run_in_thread(
            func=self._blocking_stop_data_stream, timeout=COMMUNICATION_TIMEOUT
        )

    def _blocking_connect(self) -> None:
        """Connect and then read the specified channels.

        This makes sure that the configured channels can be read.
        """
        super()._blocking_connect()

        # Read each input channel, to make sure the configuration is valid
        input_channels = [f"AIN{i}" for i in self.config.analog_inputs]
        num_frames = len(input_channels)
        values = ljm.eReadNames(self.handle, num_frames, input_channels)
        if len(values) != len(input_channels):
            raise RuntimeError(
                f"len(input_channels)={input_channels} != len(values)={values}"
            )
