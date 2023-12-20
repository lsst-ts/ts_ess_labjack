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
import contextlib
import logging
import pathlib
import types
import unittest
from collections.abc import AsyncGenerator
from typing import TypeAlias

import numpy as np
import pytest
import yaml
from lsst.ts import salobj
from lsst.ts.ess import common, labjack

logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s", level=logging.DEBUG
)

PathT: TypeAlias = str | pathlib.Path

# Standard timeout in seconds.
TIMEOUT = 5

# Random generator for mock data.
RNG = np.random.default_rng(42)

# Location string.
LOCATION_STRING = "none, here"


class DataClientTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.log = logging.getLogger()
        self.data_dir = (
            pathlib.Path(__file__).parent / "data" / "config" / "data_client"
        )

        config_schema = labjack.LabJackDataClient.get_config_schema()
        self.validator = salobj.DefaultingValidator(config_schema)

    @contextlib.asynccontextmanager
    async def make_topics(self) -> AsyncGenerator[types.SimpleNamespace, None]:
        if hasattr(salobj, "set_random_topic_subname"):
            salobj.set_random_topic_subname()
        else:
            salobj.set_random_lsst_dds_partition_prefix()
        async with salobj.make_mock_write_topics(
            name="ESS",
            attr_names=["tel_airTurbulence", "tel_pressure", "evt_sensorStatus"],
        ) as topics:
            yield topics

    async def test_constructor_good_full(self) -> None:
        """Construct with good_full.yaml and compare values to that file.

        Use the default simulation_mode.
        """
        async with self.make_topics() as topics:
            config = self.get_config("good_full.yaml")
            data_client = labjack.LabJackDataClient(
                config=config,
                topics=topics,
                log=self.log,
            )
            assert data_client.simulation_mode == 0
            assert isinstance(data_client.log, logging.Logger)
            assert set(data_client.channel_names) == {f"AIN{i}" for i in (0, 1)}
            assert len(data_client.channel_names) == len(data_client.offsets)
            assert len(data_client.channel_names) == len(data_client.scales)
            assert isinstance(data_client.processor, common.processor.BaseProcessor)

    async def test_constructor_good_minimal(self) -> None:
        """Construct with good_minimal.yaml and compare values to that file.

        Use the default simulation_mode.
        """
        async with self.make_topics() as topics:
            config = self.get_config("good_minimal.yaml")
            data_client = labjack.LabJackDataClient(
                config=config,
                topics=topics,
                log=self.log,
            )
            assert data_client.simulation_mode == 0
            assert isinstance(data_client.log, logging.Logger)
            assert set(data_client.channel_names) == {f"AIN{i}" for i in (0, 1)}
            assert len(data_client.channel_names) == len(data_client.offsets)
            assert len(data_client.channel_names) == len(data_client.scales)
            assert isinstance(data_client.processor, common.processor.BaseProcessor)

    async def test_constructor_specify_simulation_mode(self) -> None:
        async with self.make_topics() as topics:
            config = self.get_config("good_minimal.yaml")
            for simulation_mode in (0, 1):
                data_client = labjack.LabJackDataClient(
                    config=config,
                    topics=topics,
                    log=self.log,
                    simulation_mode=simulation_mode,
                )
                assert data_client.simulation_mode == simulation_mode

    async def test_auxtel_camera_coolant_pressure(self) -> None:
        async with self.make_topics() as topics:
            config = self.get_config("auxtel_camera_coolant_pressure.yaml")
            data_client = labjack.LabJackDataClient(
                config=config, topics=topics, log=self.log, simulation_mode=1
            )
            assert data_client.handle is None
            assert data_client.run_task.done()

            num_channels = 2
            data_client.mock_raw_data = RNG.random(num_channels)
            converted_values = []
            # Apply the corresponding offset and scale to each value.
            for i in range(len(data_client.mock_raw_data)):
                converted_values.append(
                    (data_client.mock_raw_data[i] - config.offsets[i])
                    * config.scales[i]
                )

            await data_client.start()
            assert data_client.handle is not None
            assert not data_client.run_task.done()

            # Wait for data to be written, then check it.
            data_client.wrote_event.clear()
            await asyncio.wait_for(data_client.wrote_event.wait(), timeout=TIMEOUT)
            await data_client.stop()
            assert data_client.handle is None
            assert data_client.run_task.done()

            assert topics.tel_pressure.data.sensorName == config.sensor_name
            for i in range(len(converted_values)):
                assert converted_values[i] == pytest.approx(
                    topics.tel_pressure.data.pressureItem[i]
                )
            self.check_event(topics.evt_sensorStatus, config.sensor_name, 0, 0)

    async def test_registry(self) -> None:
        data_client_class = common.data_client.get_data_client_class(
            "LabJackDataClient"
        )
        assert data_client_class is labjack.LabJackDataClient

    def check_event(
        self,
        topic: types.SimpleNamespace,
        sensor_name: str,
        sensor_status: int,
        server_status: int,
    ) -> None:
        """Check event values.

        Parameters
        ----------
        topic : types.SimpleNamespace
            The topic.
        sensor_name : `str`
            The sensor name.
        sensor_status : `int`
            The sensor status.
        server_status : `int`
            The server status.
        """
        assert topic.data.sensorName == sensor_name
        assert topic.data.sensorStatus == sensor_status
        assert topic.data.serverStatus == server_status

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
