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
import unittest
import types
from typing import TypeAlias

import numpy as np
import pytest
import yaml

from lsst.ts import salobj
from lsst.ts.ess import common
from lsst.ts.ess import labjack

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

        Since the PSDs are random, don't try to check the PSD data.
        """
        config = self.get_config("good_full_two_accelerometers.yaml")
        data_client = labjack.LabJackAccelerometerDataClient(
            config=config, topics=self.topics, log=self.log, simulation_mode=1
        )
        assert data_client.handle is None
        assert data_client.run_task.done()

        await data_client.start()
        assert data_client.handle is not None
        assert not data_client.run_task.done()

        # Wait for data to be written, then check the raw values
        # (the raw data is random noise, so don't bother to check the
        # the computed PSD data; see test_power_spectral_density for that).
        data_client.wrote_event.clear()
        await asyncio.wait_for(data_client.wrote_event.wait(), timeout=TIMEOUT)

        assert set(data_client.psd_topic.data_dict.keys()) == {"alpha", "beta"}
        for data in data_client.psd_topic.data_dict.values():
            assert data.numDataPoints == data_client.config.num_frequencies
            assert (
                data_client.psd_frequencies[-data_client.config.num_frequencies - 1]
                <= data_client.config.min_frequency
                <= data_client.psd_frequencies[-data_client.config.num_frequencies]
            )
            assert data.maxPSDFrequency == pytest.approx(
                data_client.config.max_frequency
            )
            assert len(data.accelerationPSDX) == 200

        # Wait for data to be written, then check the raw data.
        # With the configured values there are 4
        # Only the last batch of values is saved by the mock topic.
        assert data_client.accel_array_len == 200
        assert data_client.num_samples == 394
        start_accel_index = 200
        num_accel_values = 194
        scaled_data = data_client.scaled_data_from_raw(data_client.mock_raw_1d_data)
        scaled_row_index = 0
        for sensor_name, accel_data in data_client.accel_topic.data_dict.items():
            assert accel_data.sensorName == sensor_name
            assert accel_data.numDataPoints == num_accel_values
            assert len(accel_data.accelerationX) == 200
            for axis in ("X", "Y", "Z"):
                field_name = f"acceleration{axis}"
                accel_array = getattr(accel_data, field_name)
                assert np.allclose(
                    scaled_data[
                        scaled_row_index,
                        start_accel_index : start_accel_index + num_accel_values,
                    ],
                    accel_array[0:num_accel_values],
                )
                scaled_row_index += 1

        data_client.wrote_event.clear()
        await asyncio.wait_for(data_client.wrote_event.wait(), timeout=TIMEOUT)

        await data_client.stop()
        assert data_client.handle is None
        assert data_client.run_task.done()

    async def test_power_spectral_density(self) -> None:
        """Test operation with expected PSDs."""
        config = self.get_config("good_full_one_accelerometer.yaml")
        data_client = labjack.LabJackAccelerometerDataClient(
            config=config, topics=self.topics, log=self.log, simulation_mode=1
        )
        # Wait to output data until we provide our own mock_raw_1d_data.
        data_client.make_random_mock_raw_1d_data = False
        assert data_client.handle is None
        assert data_client.run_task.done()

        # start the data client and wait for it to be configured
        await data_client.start()
        assert data_client.handle is not None
        for _ in range(10):
            if data_client.sampling_interval is not None:
                break
            await asyncio.sleep(0.1)
        else:
            assert False, "Timed out waiting for sampling_interval to be set"

        # These values are all based on the exact configuration
        # in good_full_one_accelerometer.yaml, which were chosen to give
        # rounded values for PSD frequencies and an exact match
        # between one of those frequencies and config.min_frequency.
        assert data_client.psd_start_index == 4
        assert data_client.num_samples == 200
        assert np.allclose(
            data_client.psd_frequencies, np.linspace(start=0, stop=500, num=101)
        )
        num_samples = data_client.num_samples
        # Generate mock raw data.
        # Pick an arbitrary collection of frequencies for each axis.
        # In order to simplify the code and interpretation of the result,
        # make each frequency match one of the frequencies.
        # of the reported PSD and make them all the same amplitude.
        axis_frequencies: dict[str, list[float]] = dict()
        assert (
            data_client.psd_frequencies[-data_client.config.num_frequencies - 1]
            <= data_client.config.min_frequency
            <= data_client.psd_frequencies[-data_client.config.num_frequencies]
        )
        assert data_client.config.max_frequency == pytest.approx(
            data_client.psd_frequencies[-1]
        )

        raw_2d_data = np.zeros(shape=(3, num_samples))
        # Dict of axis: list of frequency indices
        # where each index is relative to the full frequency array
        # (rather than the subset published).
        frequency_indices = dict(
            X=[8, 12, 60],
            Y=[9, 17],
            Z=[40],
        )
        for axis, indices in frequency_indices.items():
            assert (
                np.min(indices) >= data_client.psd_start_index
            ), f"all {indices=} must be ≥ {data_client.psd_start_index=}"
            axis_frequencies[axis] = [
                data_client.psd_frequencies[index] for index in indices
            ]
        time_array = np.arange(
            start=0,
            stop=data_client.sampling_interval * (num_samples - 0.1),
            step=data_client.sampling_interval,
        )
        assert len(time_array) == num_samples
        for i, (axis, frequencies) in enumerate(axis_frequencies.items()):
            arrays = [
                np.sin(time_array * np.pi * 2 * frequency) for frequency in frequencies
            ]
            raw_2d_data[i] = np.sum(arrays, axis=0)
        raw_1d_data = np.reshape(raw_2d_data, newshape=(num_samples * 3), order="F")
        assert raw_1d_data[0] == raw_2d_data[0, 0]
        assert raw_1d_data[1] == raw_2d_data[1, 0]
        assert raw_1d_data[2] == raw_2d_data[2, 0]
        assert raw_1d_data[3] == raw_2d_data[0, 1]
        assert raw_1d_data[4] == raw_2d_data[1, 1]
        assert raw_1d_data[5] == raw_2d_data[2, 1]

        data_client.mock_raw_1d_data = raw_1d_data

        # Wait for data to be written, then check it.
        data_client.wrote_event.clear()
        await asyncio.wait_for(data_client.wrote_event.wait(), timeout=TIMEOUT)

        assert not data_client.config.write_acceleration
        assert data_client.accel_topic.data_dict == {}

        psd_data = data_client.psd_topic.data
        assert psd_data.sensorName == "alpha"
        assert psd_data.numDataPoints == data_client.config.num_frequencies
        assert psd_data.minPSDFrequency == pytest.approx(
            data_client.config.min_frequency
        )
        assert psd_data.maxPSDFrequency == pytest.approx(
            data_client.config.max_frequency
        )
        assert len(psd_data.accelerationPSDX) == 200
        for axis_index, (axis, indices) in enumerate(frequency_indices.items()):
            scale = data_client.accelerometers[0].scales[axis_index]
            field_name = f"accelerationPSD{axis}"
            psd = getattr(psd_data, field_name)
            assert np.all(np.isnan(psd[psd_data.numDataPoints :]))
            ideal_psd = np.zeros(psd_data.numDataPoints)
            for index in indices:
                # The value 1e4 is based on observation.
                ideal_psd[index - data_client.psd_start_index] = 1e4 * scale**2
            assert np.allclose(psd[0 : psd_data.numDataPoints], ideal_psd)

        await data_client.stop()
        assert data_client.handle is None
        assert data_client.run_task.done()

    async def test_registry(self) -> None:
        data_client_class = common.get_data_client_class(
            "LabJackAccelerometerDataClient"
        )
        assert data_client_class is labjack.LabJackAccelerometerDataClient

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
