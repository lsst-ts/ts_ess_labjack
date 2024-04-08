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
import collections.abc
import itertools
import logging
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

# Maximum frequency (Hz) at which the mock simulator can read an input.
# The max read frequency per channel = MAX_MOCK_READ_FREQUENCY / num channels.
MAX_MOCK_READ_FREQUENCY = 10000

# Number of channels per accelerometer
# We assume a 3-axis accelerometr: x, y, z
NUM_CHANNELS_PER_ACCELEROMETER = 3

# Number of accelerometer arrays full of random data to auto-generate.
# 5 gives a few unique accelerometer messages to cycle through,
# without using a crazy amount of memory.
NUM_RANDOM_ACCEL_ARRAYS = 5

# Random generator for mock data.
RNG = np.random.default_rng(42)


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

    Attributes
    ----------
    accel_array_len : `int`
        The number of raw samples in each accelerometer telemetry array.
    accelerometers : `list` [`types.SimpleNamespace`]
        List of configuration for each accelerometer.
    acquisition_time : `float` | `None`
        How long it takes to acquire one set of raw data samples*:
        sampling_interval * (accel_array_len - 1)
    num_channels : `int`
        Number of accelerometer channels to read (3 per accelerometer).
    psd_array_len : `int`
        The number of samples in each accelerometerPSD telemetry array.
    psd_frequencies : `np.ndarray`
        Array of frequences for the computed PSD, starting from 0.
    sampling_interval : `float` | `None`
        Interval between samples for one channel (seconds)*.
    offsets : `ndarray`
        Offset for each accelerometer channel.
        scaled acceleration = (raw acceleration - offset) * scale
    scales : `ndarray`
        Scale for each accelerometer channel; see ``offsets`` for details.
    make_random_mock_raw_1d_data: `bool`
        In simulation mode controls whether the client generates random
        mock_raw_1d_data (producing garbage PSDs).
        By default it is True, so that some output is produced.
        Set it to False if you would rather generate the raw data yourself,
        in which case the topic is not written until you set
        ``mock_raw_1d_data``.
    mock_raw_1d_data: `list` [`float`] | `None`
        Unit tests may set this to a list of floats with length
        containing raw acceleration values for:
        [v0_chan0, v0_chan1, v0_chan2, v1_chan0, v1_chan1, v1_chan2,
        ..., vn_chan0, vn_chan1, vn_chan2].
        In simulation mode this will be cycled over to provide the
        raw acceleration data.
        If None and if make_random_mock_raw_1d_data is true
        then it will be set to random data when first needed.
    wrote_psd_event: `asyncio.Event`
        An event that unit tests can use to wait for PSD data to be written.
        A test can clear the event, then wait for it to be set.

    Notes
    -----
    In simulation mode the mock LabJack returns unspecified values,
    and those values may change in future versions of the LabJack software.

    Attributes marked with * are set when sampling begins,
    because the exact values depend on a value returned by the LabJack.
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

        self.accel_topic = topics.tel_accelerometer
        self.psd_topic = topics.tel_accelerometerPSD

        # Length of the array fields in the accelerometer topic.
        self.accel_array_len = len(self.accel_topic.DataType().accelerationX)
        # Length of the array fields in the accelerometerPSD topic.
        self.psd_array_len = len(self.psd_topic.DataType().accelerationPSDX)

        if self.accel_array_len != 2 * self.psd_array_len - 2:
            raise RuntimeError(
                f"num accel points = {self.accel_array_len} != "
                f"num PSD points = {self.psd_array_len} * 2 - 2"
            )

        self.accelerometers = [
            types.SimpleNamespace(**accel_dict)
            for accel_dict in self.config.accelerometers
        ]

        self.num_channels = NUM_CHANNELS_PER_ACCELEROMETER * len(
            self.config.accelerometers
        )

        self.psd_frequencies: np.ndarray | None = None

        self.sampling_interval: float | None = None
        self.acquisition_time: float | None = None

        # Set modbus address, {offset}, and scale for every
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

        self.loop = asyncio.get_running_loop()
        self.mock_stream_task = utils.make_done_future()

        # A task that monitors self.process_data
        self.process_data_task = utils.make_done_future()

        self.make_random_mock_raw_1d_data = True
        self.mock_raw_1d_data: list[float] | None = None

        self.wrote_psd_event = asyncio.Event()

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        return yaml.safe_load(
            """
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
  max_frequency:
    description: >-
        Maximum frequency to report in the PSD (Hz).
        If larger than the LabJack can handle then it is reduced.
    type: number
    minimum: 0
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
  - max_frequency
  - accelerometers
additionalProperties: false
"""
        )

    async def run(self) -> None:
        """Read and process data from the LabJack."""
        await self.start_data_stream()
        await asyncio.Future()

    async def read_data(self) -> None:
        """Read data.

        Notes
        -----
        This method should never be called because this class overrides the
        run method which, in the super class, calls this read_data method.
        """
        raise NotImplementedError(
            "This method should never be called. Please ensure that "
            "the `run` method is implemented in this class."
        )

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

        Stream self.mock_raw_1d_data, cycling through it forever.
        If self.mock_raw_1d_data is None and
        self.make_random_mock_raw_1d_data is true (the default), set it to
        a random array long enough for NUM_RANDOM_ACCEL_ARRAYS raw reads.
        """
        self.log.info("mock_stream begins")
        try:
            if self.acquisition_time is None:
                raise RuntimeError("Sampling has not been configured.")

            sleep_interval = self.acquisition_time
            mock_raw_iter: collections.abc.Iterator[float] | None = None
            num_raw_samples = (
                NUM_RANDOM_ACCEL_ARRAYS * self.accel_array_len * self.num_channels
            )
            if self.mock_raw_1d_data is None and self.make_random_mock_raw_1d_data:
                # Generate random mock data
                # using half the available scale of -10 to 10 volts.
                self.mock_raw_1d_data = list(RNG.random(num_raw_samples) * 10 - 5)
            while True:
                end_tai: float | None = None
                scaled_data: np.ndarray | None = None
                await asyncio.sleep(sleep_interval)
                if self.mock_raw_1d_data is not None:
                    if mock_raw_iter is None:
                        mock_raw_iter = itertools.cycle(self.mock_raw_1d_data)
                    next_raw_arr = [
                        next(mock_raw_iter)
                        for _ in range(self.accel_array_len * self.num_channels)
                    ]
                    end_tai = utils.current_tai()
                    scaled_data = self.scaled_data_from_raw(next_raw_arr)
                if scaled_data is not None and end_tai is not None:
                    self.start_processing_data(scaled_data, end_tai, (0, 0))
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

        # The desired frequency at which to acquire data for a channel.
        desired_scan_frequency = 2 * self.config.max_frequency

        # LabJack ljm demo mode does not support streaming,
        # so use mock streaming.
        if self.simulation_mode == 0:
            actual_scan_frequency = ljm.eStreamStart(
                self.handle,
                self.accel_array_len,
                len(self.modbus_addresses),
                self.modbus_addresses,
                desired_scan_frequency,
            )
            ljm.setStreamCallback(self.handle, self.blocking_data_stream_callback)
        else:
            actual_scan_frequency = min(
                desired_scan_frequency, MAX_MOCK_READ_FREQUENCY / self.num_channels
            )
            self.loop.call_soon_threadsafe(self.start_mock_stream)

        # Do not await until self.sampling_interval, self.acquisition_time,
        # and self.psd_frequencies have all been computed.
        # The background task started just above needs them.

        self.sampling_interval = 1 / actual_scan_frequency
        assert self.sampling_interval is not None  # mypy idiocy
        self.acquisition_time = self.sampling_interval * (self.accel_array_len - 1)

        self.log.info(f"{desired_scan_frequency=}, {actual_scan_frequency=}")
        # Warn if the LabJack cannot gather data as quickly as requested.
        # Allow a bit of margin for roundoff error (the log statement
        # above may help determine a good value for this margin).
        if actual_scan_frequency < desired_scan_frequency * 0.99:
            actual_psd_max_frequency = actual_scan_frequency / 2
            self.log.warning(
                "LabJack cannot gather data that quickly; "
                f"{self.config.max_frequency=} reduced to {actual_psd_max_frequency}"
            )

        # Compute self.psd_frequencies
        self.psd_frequencies = np.fft.rfftfreq(
            self.accel_array_len, self.sampling_interval
        )
        assert self.psd_frequencies is not None  # make mypy happy
        assert len(self.psd_frequencies) == self.psd_array_len

        self.log.info(f"actual max_frequency={self.psd_frequencies[-1]:0.2f}")

    def blocking_data_stream_callback(self, handle: int) -> None:
        """Called in a thread when a full set of  stream data is available.

        A full set is self.accel_array_len * self.num_channels values.
        """
        end_tai = utils.current_tai()
        try:
            (
                raw_1d_data,
                backlog1,
                backlog2,
            ) = ljm.eStreamRead(self.handle)
            scaled_data = self.scaled_data_from_raw(raw_1d_data)
            self.loop.call_soon_threadsafe(
                self.start_processing_data, scaled_data, end_tai, (backlog1, backlog2)
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
            np.abs(
                np.fft.rfft(scaled_data)  # type: ignore
                * self.sampling_interval  # type: ignore
                / self.accel_array_len  # type: ignore
            )
            ** 2
        )

    def start_processing_data(
        self, scaled_data: np.ndarray, end_tai: float, backlogs: tuple[int, int]
    ) -> None:
        """Start process_data as a background task, unless still running.

        Parameters
        ----------
        scaled_data : `np.ndarray`
            Acceleration data of shape (self.num_channels,
            self.accel_array_len) after applying offset and scale.
        end_tai : `float`
            The time (TAI unix seconds) at which data collection ended.
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
            self.process_data(
                scaled_data=scaled_data, end_tai=end_tai, backlogs=backlogs
            )
        )

    async def process_data(
        self, scaled_data: np.ndarray, end_tai: float, backlogs: tuple[int, int]
    ) -> None:
        """Process one set of data.

        Parameters
        ----------
        scaled_data : `np.ndarray`
            x, y, z acceleration data of shape (self.num_channels, n)
            after applying offset and scale
        end_tai : `float`
            The time (TAI unix seconds) at which data collection ended.
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
        if (
            self.sampling_interval is None
            or self.acquisition_time is None
            or self.psd_frequencies is None
        ):
            raise RuntimeError("Sampling has not been configured")

        if scaled_data.shape != (self.num_channels, self.accel_array_len):
            self.log.error(
                f"Bug: {scaled_data.shape=}[0] != ({self.num_channels=}, {self.accel_array_len=}); "
                "callback ignoring this data"
            )
            return

        start_tai = end_tai - self.acquisition_time

        try:
            for accel_index, accelerometer in enumerate(self.accelerometers):
                channel_start_index = accel_index * NUM_CHANNELS_PER_ACCELEROMETER
                accel_kwargs = {
                    f"acceleration{axis}": scaled_data[
                        channel_start_index + channel_offset, :
                    ]
                    for channel_offset, axis in enumerate(("X", "Y", "Z"))
                }
                await self.accel_topic.set_write(
                    sensorName=accelerometer.sensor_name,
                    timestamp=start_tai,
                    location=accelerometer.location,
                    interval=self.sampling_interval,
                    **accel_kwargs,
                )

            psd = self.psd_from_scaled_data(scaled_data=scaled_data)
            for accel_index, accelerometer in enumerate(self.accelerometers):
                channel_start_index = accel_index * NUM_CHANNELS_PER_ACCELEROMETER
                psd_kwargs = {
                    f"accelerationPSD{axis}": psd[
                        channel_start_index + channel_offset, :
                    ]
                    for channel_offset, axis in enumerate(("X", "Y", "Z"))
                }
                await self.psd_topic.set_write(
                    sensorName=accelerometer.sensor_name,
                    timestamp=start_tai,
                    location=accelerometer.location,
                    maxPSDFrequency=self.psd_frequencies[-1],
                    **psd_kwargs,
                )
            self.wrote_psd_event.set()
        except Exception as e:
            self.log.exception(f"process_data failed: {e!r}")
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
