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

import numpy as np
import yaml

# Hide mypy error `Module "labjack" has no attribute "ljm"`.
from labjack import ljm  # type: ignore
from lsst.ts import salobj, utils

from .base_labjack_data_client import BaseLabJackDataClient

# Time limit for communicating with the LabJack (seconds).
COMMUNICATION_TIMEOUT = 5

# Time limit for configuring streaming;
# measured time from far away is 6 seconds.
START_STREAMING_TIMEOUT = 15

# Maximum sampling frequency (Hz) that the simulator allows.
MAX_MOCK_SAMPLE_FREQUENCY = 1000

# Smallest allowed max_frequency / config.min_frequency.
MIN_FREQUENCY_RATIO = 2

# Number of channels per accelerometer
# We assume a 3-axis accelerometr: x, y, z
NUM_CHANNELS_PER_ACCELEROMETER = 3


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

        self.psd_topic = topics.tel_accelerometerPSD
        self.accel_topic = topics.tel_accelerometer

        self.psd_array_len = len(self.psd_topic.DataType().accelerationPSDX)
        self.accel_array_len = len(self.accel_topic.DataType().accelerationX)

        if config.min_frequency * MIN_FREQUENCY_RATIO > config.max_frequency:
            raise ValueError(
                f"{self.config.min_frequency=} must be < "
                f"{config.max_frequency=} / {MIN_FREQUENCY_RATIO=} "
                f"= {config.max_frequency / MIN_FREQUENCY_RATIO}"
            )

        # List of accelerometer config as simple namespaces (structs);
        # this is easier to work with self.config.accelerometers,
        # which is a list of dicts.
        self.accelerometers = [
            types.SimpleNamespace(**accel_dict)
            for accel_dict in self.config.accelerometers
        ]

        # Interval between samples (seconds).
        # Set by _blocking_start_data_stream
        # since the LabJack may offer a smaller value than requested.
        self.sampling_interval: None | float = None

        # Number of samples (per channel) to measure PSD.
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

        self.num_channels = NUM_CHANNELS_PER_ACCELEROMETER * len(
            self.config.accelerometers
        )

        # Set modbus address, offset, and scale for every
        # LabJack analog input channel to read, in order:
        # accel 0 x, accel 0 y, accel 0 z, accel 1 x, accel 1 y, ...
        # Note: The modbus address for analog input "AIN{n}" is 2*n
        self.modbus_addresses: list[int] = []
        offsets: list[float] = []
        scales: list[float] = []
        for accelerometer in self.accelerometers:
            self.modbus_addresses += [
                analog_input * 2 for analog_input in accelerometer.analog_inputs
            ]
            offsets += list(accelerometer.offsets)
            scales += list(accelerometer.scales)
        self.offsets = np.array(offsets)
        self.scales = np.array(scales)

        if len(set(self.modbus_addresses)) != len(self.modbus_addresses):
            raise ValueError("configured input_channels contain one or more duplicates")

        # Set output arrays to NaN. This is not strictly necessary
        # but makes it easier to see unset array elements.
        self.psd_topic.set(
            **{
                f"accelerationPSD{axis}": [math.nan] * self.psd_array_len
                for axis in ("X", "Y", "Z")
            },
        )

        self.loop = asyncio.get_running_loop()
        self.mock_stream_task = utils.make_done_future()

        # A task that monitors self.process_data
        self.process_data_task = utils.make_done_future()

        # In simulation mode controls whether the client generates
        # random mock_raw_1d_data (producing garbage PSDs).
        # By default it is True, so that some output is produced.
        # Set it to False if you would rather generate the raw data yourself,
        # in which case the topic is not written until
        # you set self.mock_raw_1d_data.
        self.make_random_mock_raw_1d_data = True
        # Unit tests may set this to a list of floats
        # with length 3 * self.num_samples containing:
        # [v0_chan0, v0_chan1, v0_chan2, v1_chan0, v1_chan1, v1_chan2,
        # ..., vn_chan0, vn_chan1, vn_chan2].
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
  write_acceleration:
    description: Write the acceleration telemetry topic (calibrated input data)?
    type: boolean
  accelerometers:
    description: >-
      Configuration for each 3-axis accelerometer.
      Note that the ESS will output separate accelerometerPSD messages
      for each accelerometer (each with a different value of sensorName).
    type: array
    minItems: 1
    items:
      type: object
      properties:
        sensor_name:
            description: Value for the sensorName field of the topic.
            type: string
        location:
            description: Location of sensor.
            type: string
        analog_inputs:
            description: >-
                LabJack analog inputs read for x, y, and z data.
                0 for AIN0, 1 for AIN1, 2 or AIN2, etc.
            type: array
            minItems: 3
            maxItems: 3
            items:
                type: integer
                minimum: 0
        offsets:
            description: >-
                Accelerometer offsets for x, y, and z data (Volts).
                Acceleration in m/s2 = (raw - offset) * scale
            type: array
            minItems: 3
            maxItems: 3
            items:
                type: number
        scales:
            description: >-
                Accelerometer scales for x, y, and z data (m/s2 per Volt).
                See offset for more information.
            type: array
            minItems: 3
            maxItems: 3
            items:
                type: number
      required:
        - sensor_name
        - location
        - analog_inputs
        - offsets
        - scales
      additionalProperties: false
required:
  - device_type
  - connection_type
  - identifier
  - min_frequency
  - max_frequency
  - num_frequencies
  - write_acceleration
  - accelerometers
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
        self.process_data_task.cancel()
        await super().disconnect()

    async def start_data_stream(self) -> None:
        """Start the data stream from the LabJack."""

        t0 = utils.current_tai()
        await self.run_in_thread(
            func=self._blocking_start_data_stream, timeout=START_STREAMING_TIMEOUT
        )
        dt = utils.current_tai() - t0
        self.log.debug(f"start_data_stream took {dt:0.2f} seconds")

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
                # using half the available scale of -10 to 10 volts.
                self.mock_raw_1d_data = list(
                    np.random.random(self.num_samples * self.num_channels) * 10 - 5
                )
            while True:
                await asyncio.sleep(sleep_interval)
                if self.mock_raw_1d_data is not None:
                    scaled_data = self.scaled_data_from_raw(self.mock_raw_1d_data)
                self.start_processing_data(scaled_data, (0, 0))
        except asyncio.CancelledError:
            self.log.info("mock_stream ends")
        except Exception:
            self.log.exception("mock_stream failed")
            raise

    def _blocking_start_data_stream(self) -> None:
        """Start streaming data from the LabJack.

        Call in a thread to avoid blocking the event loop.
        """
        self.mock_stream_task.cancel()

        desired_sampling_frequency = 2 * self.config.max_frequency

        # LabJack ljm demo mode does not support streaming,
        # so use mock streaming.
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
            # above may help determine a good value for this margin).
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

        # Compute self.psd_frequencies and self.psd_start_index.
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

    def blocking_data_stream_callback(self, handle: int) -> None:
        """Called in a thread when a full set of stream data is available."""
        try:
            (
                raw_1d_data,
                backlog1,
                backlog2,
            ) = ljm.eStreamRead(self.handle)
            scaled_data = self.scaled_data_from_raw(raw_1d_data)
            self.loop.call_soon_threadsafe(
                self.start_processing_data, scaled_data, (backlog1, backlog2)
            )
        except Exception as e:
            self.log.error(f"blocking_data_stream_callback failed: {e!r}")

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
            Raw data reshaped to size (self.num_channels, voltages)
            with offsets and scales applied::

                scaled = (raw - offset) * scaled
        """
        npoints = len(raw_1d_data) // self.num_channels
        raw_2d_data = np.reshape(
            raw_1d_data,
            newshape=(self.num_channels, npoints),
            order="F",
        )
        return (raw_2d_data - self.offsets[:, np.newaxis]) * self.scales[:, np.newaxis]

    def psd_from_scaled_data(self, scaled_data: np.ndarray) -> np.ndarray:
        """Compute the PSD from scaled data."""
        return (
            np.abs(np.fft.rfft(scaled_data) * self.sampling_interval / self.num_samples)
            ** 2
        )

    def start_processing_data(
        self, scaled_data: np.ndarray, backlogs: tuple[int, int]
    ) -> None:
        """Start process_data as a background task, unless still running.

        Parameters
        ----------
        scaled_data : `np.ndarray`
            x, y, z acceleration data of shape (self.num_channels, n)
            after applying offset and scale
        backlogs : `tuple`
            Two measures of the number of backlogged messages.
            Both values should be nearly zero if the data client
            is keeping up with the LabJack.
        """
        if not self.process_data_task.done():
            self.log.warning(
                "An older process_data background task is still running; skipping this data"
            )
        self.process_data_task = asyncio.create_task(
            self.process_data(scaled_data=scaled_data, backlogs=backlogs)
        )

    async def process_data(
        self, scaled_data: np.ndarray, backlogs: tuple[int, int]
    ) -> None:
        """Process one set of data.

        Parameters
        ----------
        scaled_data : `np.ndarray`
            x, y, z acceleration data of shape (self.num_channels, n)
            after applying offset and scale
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
            psd = self.psd_from_scaled_data(scaled_data)
            for i, accelerometer in enumerate(self.accelerometers):
                channel_start_index = i * NUM_CHANNELS_PER_ACCELEROMETER

                if self.config.write_acceleration:
                    # Write scaled acceleration data. If there is more data
                    # than fits in the arrays, write the data as multiple
                    # messages.
                    accel_data_start_index = 0
                    done_writing_accel = False
                    while not done_writing_accel:
                        num_accel_to_write = self.num_samples - accel_data_start_index
                        if num_accel_to_write > self.accel_array_len:
                            num_accel_to_write = self.accel_array_len
                        else:
                            done_writing_accel = True

                        accel_kwargs = {
                            f"acceleration{axis}": scaled_data[
                                channel_start_index + channel_offset,
                                accel_data_start_index : accel_data_start_index
                                + num_accel_to_write,
                            ]
                            for channel_offset, axis in enumerate(("X", "Y", "Z"))
                        }
                        accel_data_start_index += num_accel_to_write
                        await self.accel_topic.set_write(
                            sensorName=accelerometer.sensor_name,
                            location=accelerometer.location,
                            interval=self.sampling_interval,
                            numDataPoints=num_accel_to_write,
                            **accel_kwargs,
                        )

                # Write PSD data.
                psd_kwargs = {
                    f"accelerationPSD{axis}": psd[
                        channel_start_index + channel_offset, self.psd_start_index :
                    ]
                    for channel_offset, axis in enumerate(("X", "Y", "Z"))
                }
                await self.psd_topic.set_write(
                    sensorName=accelerometer.sensor_name,
                    location=accelerometer.location,
                    interval=self.sampling_interval,
                    minPSDFrequency=self.psd_frequencies[self.psd_start_index],
                    maxPSDFrequency=self.psd_frequencies[-1],
                    numDataPoints=self.config.num_frequencies,
                    **psd_kwargs,
                )
            self.wrote_event.set()
        except Exception:
            self.log.exception("process_data failed")
            raise

    def _blocking_connect(self) -> None:
        """Connect and then read the specified channels.

        This makes sure that the configured channels can be read.
        """
        super()._blocking_connect()

        # Read each input channel, to make sure the configuration is valid.
        input_channel_names = [f"AIN{addr//2}" for addr in self.modbus_addresses]
        num_frames = len(input_channel_names)
        values = ljm.eReadNames(self.handle, num_frames, input_channel_names)
        if len(values) != len(input_channel_names):
            raise RuntimeError(
                f"len(input_channel_names)={input_channel_names} != len(values)={values}"
            )
