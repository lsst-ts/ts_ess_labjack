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

TIMEOUT = 5
"""Standard timeout in seconds."""


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

        # Mock topics listed in the config files in tests/data.
        # These happen to match real array-valued ESS telemetry topics,
        # though that doesn't matter for these tests.
        self.topic = labjack.MockESSAccelerometerPSDTopic()
        self.topics = types.SimpleNamespace(**{self.topic.attr_name: self.topic})

    async def test_constructor_good_full(self) -> None:
        """Construct with good_full.yaml

        and compare values to that file.
        Use the default simulation_mode.
        """
        config = self.get_config("good_full.yaml")
        data_client = labjack.LabJackAccelerometerDataClient(
            config=config,
            topics=self.topics,
            log=self.log,
        )
        assert data_client.simulation_mode == 0
        assert isinstance(data_client.log, logging.Logger)
        assert data_client.config.analog_inputs == [0, 2, 5]
        assert data_client.modbus_addresses == [0, 4, 10]

    async def test_constructor_good_minimal(self) -> None:
        """Construct with good_minimal.yaml

        and compare values to that file.
        Use the default simulation_mode.
        """
        config = self.get_config("good_minimal.yaml")
        data_client = labjack.LabJackAccelerometerDataClient(
            config=config,
            topics=self.topics,
            log=self.log,
        )
        assert data_client.simulation_mode == 0
        assert isinstance(data_client.log, logging.Logger)
        assert data_client.config.analog_inputs == [6, 3, 1]
        assert data_client.modbus_addresses == [12, 6, 2]

    async def test_constructor_specify_simulation_mode(self) -> None:
        config = self.get_config("good_minimal.yaml")
        for simulation_mode in (0, 1):
            data_client = labjack.LabJackAccelerometerDataClient(
                config=config,
                topics=self.topics,
                log=self.log,
                simulation_mode=simulation_mode,
            )
            assert data_client.simulation_mode == simulation_mode

    async def test_basic_operation(self) -> None:
        """Test operation with random PSDs"""
        config = self.get_config("good_full.yaml")
        data_client = labjack.LabJackAccelerometerDataClient(
            config=config, topics=self.topics, log=self.log, simulation_mode=1
        )
        assert data_client.handle is None
        assert data_client.run_task.done()

        await data_client.start()
        assert data_client.handle is not None
        assert not data_client.run_task.done()

        # Wait for data to be written, then check it for plausibility
        # (we do not know what values will be written).
        data_client.wrote_event.clear()
        await asyncio.wait_for(data_client.wrote_event.wait(), timeout=TIMEOUT)

        data = data_client.topic.data
        assert data.sensorName == "alt_accel"
        assert data.numDataPoints == data_client.config.num_frequencies
        assert data.minPSDFrequency == pytest.approx(data_client.config.min_frequency)
        assert data.maxPSDFrequency == pytest.approx(data_client.config.max_frequency)
        assert len(data.accelerationPSDX) == 200

        await data_client.stop()
        assert data_client.handle is None
        assert data_client.run_task.done()

    async def test_power_spectral_density(self) -> None:
        """Test operation with expected PSDs"""
        config = self.get_config("good_full.yaml")
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
        # in good_full.yaml, which were chosen to give rounded values
        # for PSD frequencies and an exact match between one of those.
        # frequencies and config.min_frequency.
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
        assert data_client.config.min_frequency == pytest.approx(
            data_client.psd_frequencies[4]
        )
        assert data_client.config.max_frequency == pytest.approx(
            data_client.psd_frequencies[-1]
        )

        raw_2d_data = np.zeros(shape=(3, num_samples))
        # Dict of axis: list of frequency indices
        # where each index is relative to the full frequency array
        # (rather than the subset published).
        frequency_indices = dict(
            X=[4, 10, 60],
            Y=[5, 17],
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

        data = data_client.topic.data
        assert data.sensorName == "alt_accel"
        assert data.numDataPoints == data_client.config.num_frequencies
        assert data.minPSDFrequency == pytest.approx(data_client.config.min_frequency)
        assert data.maxPSDFrequency == pytest.approx(data_client.config.max_frequency)
        assert len(data.accelerationPSDX) == 200
        for axis, indices in frequency_indices.items():
            field_name = f"accelerationPSD{axis}"
            psd = getattr(data, field_name)
            assert np.all(np.isnan(psd[data.numDataPoints :]))
            ideal_psd = np.zeros(data.numDataPoints)
            for index in indices:
                # The value 1e6 is based on observation (rather than
                # a prediction based on math).
                ideal_psd[index - data_client.psd_start_index] = 1e6
            assert np.allclose(psd[0 : data.numDataPoints], ideal_psd)

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
            raw_config_dict = yaml.safe_load(f.read())
        config_dict = self.validator.validate(raw_config_dict)
        return types.SimpleNamespace(**config_dict)
