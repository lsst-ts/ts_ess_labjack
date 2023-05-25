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

import asyncio
import logging
import pathlib
import types
import unittest
from typing import TypeAlias

import numpy as np
import pytest
import yaml
from lsst.ts import salobj, utils
from lsst.ts.ess import labjack
from lsst.ts.ess.common.data_client import get_data_client_class

logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s", level=logging.DEBUG
)

PathT: TypeAlias = str | pathlib.Path

# Standard timeout in seconds.
TIMEOUT = 5


class AccelerationDataClientTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.log = logging.getLogger()
        self.data_dir = (
            pathlib.Path(__file__).parent
            / "data"
            / "config"
            / "accelerometer_data_client"
        )
        config_schema = labjack.LabJackAccelerometerDataClient.get_config_schema()
        self.validator = salobj.DefaultingValidator(config_schema)

        self.psd_topic = labjack.MockESSAccelerometerPSDTopic()
        self.raw_topic = labjack.MockESSAccelerometerTopic()
        self.topics = types.SimpleNamespace(
            **{
                self.psd_topic.attr_name: self.psd_topic,
                self.raw_topic.attr_name: self.raw_topic,
            }
        )

    def get_config(self, filename: PathT) -> types.SimpleNamespace:
        """Get a config dict from tests/data.

        This should always be a good config,
        because validation is done by the ESS CSC,
        not the data client.

        Parameters
        ----------
        filename : `str` or `pathlib.Path`
            Name of config file, including ".yaml" suffix.

        Returns
        -------
        config : types.SimpleNamespace
            The config dict.
        """
        with open(self.data_dir / filename, "r") as f:
            config_dict = yaml.safe_load(f.read())
        config_dict = self.validator.validate(config_dict)
        return types.SimpleNamespace(**config_dict)

    async def make_data_client(
        self, config: types.SimpleNamespace
    ) -> labjack.LabJackAccelerometerDataClient:
        """Create and start a LabJackAccelerometerDataClient.

        Parameters
        ----------
        config : types.SimpleNamespace
            Configuration, e.g. from `get_config`.
        """
        data_client = labjack.LabJackAccelerometerDataClient(
            config=config, topics=self.topics, log=self.log, simulation_mode=1
        )
        assert data_client.handle is None
        assert data_client.run_task.done()

        await data_client.start()
        assert data_client.handle is not None
        for _ in range(10):
            if data_client.sampling_interval is not None:
                break
            await asyncio.sleep(0.1)
        else:
            assert False, "Timed out waiting for sampling_interval to be set"

        return data_client

    def make_raw_cosines_data_and_expected_psd(
        self,
        data_client: labjack.LabJackAccelerometerDataClient,
        frequency_indices_per_channel: list[list[int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Make cos-based raw data and expected PSD data

        The raw data for each channel is the sum of zero or more cosine waves
        with amplitude 1, plus the offset specified for that channel.
        (so that the scaled data will have a mean of 0 unless
        ``frequency_indices_per_channel`` includes an entry with frequency
        index 0, in which case the scaled mean will be the scale
        for that channel).

        Parameters
        ----------
        data_client : labjack.LabJackAccelerometerDataClient
            Data client that has been started.
        frequency_indices_per_channel : list[list[int]]
            List of frequency indices, with one entry per channel.
            Each frequency index is an index into data_client.psd_frequencies.
        """
        assert len(frequency_indices_per_channel) == data_client.num_channels
        expected_psd_data = np.zeros(
            shape=(data_client.num_channels, len(data_client.psd_frequencies))
        )
        # List of cosine frequencies for each channel
        axis_frequencies_per_channel = [
            [data_client.psd_frequencies[freq_index] for freq_index in freq_indices]
            for freq_indices in frequency_indices_per_channel
        ]
        raw_2d_data = np.zeros(
            shape=(data_client.num_channels, data_client.accel_array_len)
        )
        time_array = np.arange(
            start=0,
            stop=data_client.sampling_interval * (data_client.accel_array_len - 0.1),
            step=data_client.sampling_interval,
        )
        assert len(time_array) == data_client.accel_array_len
        for channel_index, frequencies in enumerate(axis_frequencies_per_channel):
            arrays = [
                np.cos(time_array * np.pi * 2 * frequency) for frequency in frequencies
            ]
            raw_2d_data[channel_index] = np.sum(arrays, axis=0)

        # Add offsets and set non-zero elements of expected_psd_data
        for accel_index, accelerometer in enumerate(data_client.config.accelerometers):
            for axis_index in range(3):
                offset = accelerometer["offsets"][axis_index]
                scale = accelerometer["scales"][axis_index]
                channel_index = accel_index * 3 + axis_index
                raw_2d_data[channel_index] += offset

                for freq_index in frequency_indices_per_channel[channel_index]:
                    # Scale by 0.25 if a cosine wave, or 1 if
                    # input is a constant. The 0.25 is presumably related
                    # to the RMS value of a sine wave.
                    sin_factor = 1 if freq_index == 0 else 0.25
                    expected_psd_value = (
                        sin_factor * (scale * data_client.sampling_interval) ** 2
                    )
                    expected_psd_data[channel_index, freq_index] = expected_psd_value

        raw_1d_data = np.reshape(
            raw_2d_data,
            newshape=(data_client.num_channels * data_client.accel_array_len),
            order="F",
        )
        assert raw_1d_data[0] == raw_2d_data[0, 0]
        assert raw_1d_data[1] == raw_2d_data[1, 0]
        assert raw_1d_data[2] == raw_2d_data[2, 0]
        return raw_1d_data, expected_psd_data

    async def test_constructor_good_full(self) -> None:
        """Construct with good_full_two_accelerometers.yaml
        and compare values to that file.

        Use the default simulation_mode.
        """
        config = self.get_config("good_full_two_accelerometers.yaml")
        data_client = labjack.LabJackAccelerometerDataClient(
            config=config,
            topics=self.topics,
            log=self.log,
        )
        assert data_client.simulation_mode == 0
        assert isinstance(data_client.log, logging.Logger)
        assert len(data_client.config.accelerometers) == 2
        assert data_client.accelerometers[0].analog_inputs == [0, 2, 5]
        assert data_client.accelerometers[1].analog_inputs == [6, 1, 3]
        assert data_client.modbus_addresses == [0, 4, 10, 12, 2, 6]

    async def test_constructor_good_minimal(self) -> None:
        """Construct with good_minimal.yaml and compare values to that file.

        Use the default simulation_mode.
        """
        config = self.get_config("good_minimal_one_accelerometer.yaml")
        data_client = labjack.LabJackAccelerometerDataClient(
            config=config,
            topics=self.topics,
            log=self.log,
        )
        assert data_client.simulation_mode == 0
        assert isinstance(data_client.log, logging.Logger)
        assert len(data_client.config.accelerometers) == 1
        assert data_client.accelerometers[0].analog_inputs == [6, 3, 1]
        assert data_client.modbus_addresses == [12, 6, 2]

    async def test_constructor_specify_simulation_mode(self) -> None:
        config = self.get_config("good_minimal_one_accelerometer.yaml")
        for simulation_mode in (0, 1):
            data_client = labjack.LabJackAccelerometerDataClient(
                config=config,
                topics=self.topics,
                log=self.log,
                simulation_mode=simulation_mode,
            )
            assert data_client.simulation_mode == simulation_mode

    async def test_basic_operation(self) -> None:
        """Test operation with random PSDs.

        The accelerations are randomly generated in this test, so do not
        check the PSD values. PSD computation is tested in a separate test.
        """
        config = self.get_config("good_full_two_accelerometers.yaml")
        start_tai = utils.current_tai()
        data_client = await self.make_data_client(config=config)

        # Wait for 2 sets of PSD data to be written, then check raw values
        # (the raw data is random noise, so don't bother to check the
        # the computed PSD data; see test_power_spectral_density for that).
        for _ in range(2):
            data_client.wrote_psd_event.clear()
            await asyncio.wait_for(data_client.wrote_psd_event.wait(), timeout=TIMEOUT)

        assert data_client.sampling_interval == pytest.approx(0.000625)
        assert data_client.accel_array_len == 400
        assert data_client.psd_array_len == 201

        # There should be two messages for each of two accelerometers.
        assert set(data_client.accel_topic.data_dict.keys()) == {"alpha", "beta"}
        for data_list in data_client.accel_topic.data_dict.values():
            assert len(data_list) == 2
        assert set(data_client.psd_topic.data_dict.keys()) == {"alpha", "beta"}
        for data_list in data_client.psd_topic.data_dict.values():
            assert len(data_list) == 2
        last_psd_timestamps: dict[int, float] = dict()
        for data_list in data_client.psd_topic.data_dict.values():
            assert len(data_list) == 2
            for i, psd_data in enumerate(data_list):
                assert psd_data.maxPSDFrequency == pytest.approx(
                    data_client.config.max_frequency
                )
                assert len(psd_data.accelerationPSDX) == 201
                assert psd_data.timestamp > start_tai
                if i in last_psd_timestamps:
                    assert psd_data.timestamp == last_psd_timestamps[i]
                    # Also make sure time advances by the expected amount.
                    if i > 0:
                        expected_psd_read_interval = (
                            data_client.sampling_interval * data_client.accel_array_len
                        )

                        dt = psd_data.timestamp - last_psd_timestamps[i - 1]
                        assert dt == pytest.approx(expected_psd_read_interval, abs=0.05)
                else:
                    last_psd_timestamps[i] = psd_data.timestamp

        # Check the raw data.
        assert data_client.accel_array_len == 400
        assert data_client.psd_array_len == 201
        start_accel_index = 201
        scaled_data = data_client.scaled_data_from_raw(data_client.mock_raw_1d_data)
        channel_index = 0
        last_accel_timestamps: dict[int, float] = dict()
        for sensor_name, accel_data_list in data_client.accel_topic.data_dict.items():
            # There should be 2 sets data for each accelerometer
            assert len(accel_data_list) == 2
            start_accel_index = 0
            for i, accel_data in enumerate(accel_data_list):
                assert accel_data.sensorName == sensor_name
                assert accel_data.timestamp > start_tai
                if i in last_accel_timestamps:
                    assert accel_data.timestamp == last_accel_timestamps[i]
                    # Also make sure time advances by the expected amount.
                    if i > 0:
                        expected_raw_read_interval = (
                            data_client.sampling_interval * data_client.accel_array_len
                        )
                        dt = accel_data.timestamp - last_accel_timestamps[i - 1]
                        assert dt == pytest.approx(expected_raw_read_interval, abs=0.05)
                else:
                    last_accel_timestamps[i] = accel_data.timestamp
                if channel_index == 3:
                    for axis in ("X", "Y", "Z"):
                        field_name = f"acceleration{axis}"
                        accel_array = getattr(accel_data, field_name)
                        assert len(accel_array) == 400
                        assert np.allclose(
                            scaled_data[
                                channel_index,
                                start_accel_index : start_accel_index
                                + data_client.accel_array_len,
                            ],
                            accel_array[0 : data_client.accel_array_len],
                        )
                        channel_index += 1

        data_client.wrote_psd_event.clear()
        await asyncio.wait_for(data_client.wrote_psd_event.wait(), timeout=TIMEOUT)

        await data_client.stop()
        assert data_client.handle is None
        assert data_client.run_task.done()

    async def test_power_spectral_density(self) -> None:
        """Test operation with expected PSDs.

        To simplify this test use a configuration with a single
        accelerometer, so we don't have to keep track of multiple
        messages.
        """
        config = self.get_config("good_full_one_accelerometer.yaml")
        data_client = await self.make_data_client(config=config)

        assert data_client.accel_array_len == 400
        assert data_client.psd_array_len == 201
        # These values are all based on the exact configuration
        # in good_full_one_accelerometer.yaml, which were chosen to give
        # rounded values for PSD frequencies
        assert np.allclose(
            data_client.psd_frequencies, np.linspace(start=0, stop=800, num=201)
        )
        assert data_client.config.max_frequency == pytest.approx(
            data_client.psd_frequencies[-1]
        )

        # To make it easier to predict the PSD:
        # generate input data that consists of the sum of a few sine waves,
        # each with amplitude 1 and a frequency that exactly matches
        # one of the reported frequencies.
        frequency_indices_per_channel = [
            [8, 12, 60],
            [9, 17],
            [40],
        ]
        raw_1d_data, expected_psd_data = self.make_raw_cosines_data_and_expected_psd(
            data_client=data_client,
            frequency_indices_per_channel=frequency_indices_per_channel,
        )
        data_client.mock_raw_1d_data = raw_1d_data

        # Wait for 1 set of PSD data to be written, then check it.
        data_client.wrote_psd_event.clear()
        await asyncio.wait_for(data_client.wrote_psd_event.wait(), timeout=TIMEOUT)

        # There should be one message for one accelerometer.
        assert data_client.accel_topic.data_dict.keys() == {"alpha"}
        assert len(data_client.accel_topic.data_dict["alpha"]) == 1
        assert data_client.psd_topic.data_dict.keys() == {"alpha"}
        assert len(data_client.psd_topic.data_dict["alpha"]) == 1
        psd_data = data_client.psd_topic.data_dict["alpha"][0]
        assert psd_data.sensorName == "alpha"
        assert psd_data.maxPSDFrequency == pytest.approx(
            data_client.config.max_frequency
        )
        assert len(psd_data.accelerationPSDX) == 201
        for channel_index in range(len(frequency_indices_per_channel)):
            axis = ["X", "Y", "Z"][channel_index]
            field_name = f"accelerationPSD{axis}"
            psd = getattr(psd_data, field_name)
            # The factor of 0.25 is observed for a sine wave.
            # It might be related to RMS, but RMS squared is 0.5,
            # so that is not the full story.
            assert np.allclose(psd, expected_psd_data[channel_index, :])

        await data_client.stop()
        assert data_client.handle is None
        assert data_client.run_task.done()

    async def test_psd_from_scaled_data(self) -> None:
        config = self.get_config("good_full_two_accelerometers.yaml")
        data_client = await self.make_data_client(config=config)

        # Check amplitude of 0 frequency input
        scale = 5.2
        scaled_data = (
            np.ones((data_client.num_channels, data_client.accel_array_len)) * scale
        )
        psd_data = data_client.psd_from_scaled_data(scaled_data)
        ideal_psd = np.zeros((data_client.num_channels, data_client.psd_array_len))
        ideal_psd[:, 0] = (scale * data_client.sampling_interval) ** 2
        assert np.allclose(psd_data, ideal_psd)

        frequency_indices_per_channel = [
            [5, 7],
            [4],
            [11, 22, 33],
            [0, 3],
            [5, 6, 7],
            [12],
        ]
        raw_1d_data, expected_psd_data = self.make_raw_cosines_data_and_expected_psd(
            data_client=data_client,
            frequency_indices_per_channel=frequency_indices_per_channel,
        )
        scaled_data = data_client.scaled_data_from_raw(raw_1d_data)
        psd_data = data_client.psd_from_scaled_data(scaled_data)

        assert np.allclose(psd_data, expected_psd_data)

    async def test_registry(self) -> None:
        data_client_class = get_data_client_class("LabJackAccelerometerDataClient")
        assert data_client_class is labjack.LabJackAccelerometerDataClient
