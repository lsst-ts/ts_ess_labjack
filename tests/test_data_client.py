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
import math
import pathlib
import unittest
import types
from typing import Any, TypeAlias

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

# Standard timeout in seconds
TIMEOUT = 5


class DataClientTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.log = logging.getLogger()
        self.data_dir = (
            pathlib.Path(__file__).parent / "data" / "config" / "data_client"
        )

        config_schema = labjack.LabJackDataClient.get_config_schema()
        self.validator = salobj.DefaultingValidator(config_schema)

        # Mock topics listed in the config files in tests/data.
        # These happen to match real array-valued ESS telemetry topics,
        # though that doesn't matter for these tests.
        mock_topics = [
            labjack.MockEssArrayTopic(
                attr_name="tel_temperature", field_name="temperature", field_len=16
            ),
            labjack.MockEssArrayTopic(
                attr_name="tel_pressure", field_name="pressure", field_len=8
            ),
        ]
        topics_kwargs = {topic.attr_name: topic for topic in mock_topics}
        self.topics = types.SimpleNamespace(**topics_kwargs)

    async def test_constructor_good_full(self) -> None:
        """Construct with good_full.yaml and compare values to that file.

        Use the default simulation_mode.
        """
        config = self.get_config("good_full.yaml")
        data_client = labjack.LabJackDataClient(
            config=config,
            topics=self.topics,
            log=self.log,
        )
        assert data_client.simulation_mode == 0
        assert isinstance(data_client.log, logging.Logger)
        assert set(data_client.channel_names) == {f"AIN{i}" for i in (0, 2, 3, 4, 5, 6)}
        assert len(data_client.topic_handlers) == 3
        topic_handlers = list(data_client.topic_handlers.values())
        assert topic_handlers[0].topic.attr_name == "tel_temperature"
        assert topic_handlers[0].sensor_name == "labjack_test_1"
        assert topic_handlers[0].field_name == "temperature"
        assert topic_handlers[0].location == "somewhere, nowhere, somewhere else, guess"
        assert topic_handlers[0].offset == 1.5
        assert topic_handlers[0].scale == -2.1
        assert topic_handlers[0].num_channels == 4
        assert topic_handlers[0].channel_dict == {0: "AIN0", 2: "AIN3", 3: "AIN2"}

        assert topic_handlers[1].topic.attr_name == "tel_pressure"
        assert topic_handlers[1].sensor_name == "labjack_test_2"
        assert topic_handlers[1].field_name == "pressure"
        assert topic_handlers[1].location == "top of stack, bottom of stack"
        assert topic_handlers[1].offset == 0
        assert topic_handlers[1].scale == 1
        assert topic_handlers[1].num_channels == 2
        assert topic_handlers[1].channel_dict == {0: "AIN4", 1: "AIN5"}

        assert topic_handlers[2].topic.attr_name == "tel_temperature"
        assert topic_handlers[2].sensor_name == "labjack_test_3"
        assert topic_handlers[2].field_name == "temperature"
        assert topic_handlers[2].location == "none, here"
        assert topic_handlers[2].offset == 0
        assert topic_handlers[2].scale == 1
        assert topic_handlers[2].num_channels == 2
        assert topic_handlers[2].channel_dict == {1: "AIN6"}

    async def test_constructor_good_minimal(self) -> None:
        """Construct with good_minimal.yaml and compare values to that file.

        Use the default simulation_mode.
        """
        config = self.get_config("good_minimal.yaml")
        data_client = labjack.LabJackDataClient(
            config=config,
            topics=self.topics,
            log=self.log,
        )
        assert data_client.simulation_mode == 0
        assert isinstance(data_client.log, logging.Logger)
        assert set(data_client.channel_names) == {"AIN2"}
        assert len(data_client.topic_handlers) == 1
        topic_handlers = list(data_client.topic_handlers.values())
        assert topic_handlers[0].topic.attr_name == "tel_temperature"
        assert topic_handlers[0].sensor_name == "labjack_test_1"
        assert topic_handlers[0].field_name == "temperature"
        assert topic_handlers[0].location == "none, here"
        assert topic_handlers[0].offset == 0
        assert topic_handlers[0].scale == 1
        assert topic_handlers[0].num_channels == 2
        assert topic_handlers[0].channel_dict == {1: "AIN2"}

    async def test_constructor_specify_simulation_mode(self) -> None:
        config = self.get_config("good_minimal.yaml")
        for simulation_mode in (0, 1):
            data_client = labjack.LabJackDataClient(
                config=config,
                topics=self.topics,
                log=self.log,
                simulation_mode=simulation_mode,
            )
            assert data_client.simulation_mode == simulation_mode

    async def test_operation(self) -> None:
        config = self.get_config("good_full.yaml")
        data_client = labjack.LabJackDataClient(
            config=config, topics=self.topics, log=self.log, simulation_mode=1
        )
        assert len(data_client.topic_handlers) == 3
        topic_handlers = list(data_client.topic_handlers.values())
        assert data_client.handle is None
        assert data_client.run_task.done()

        num_channels = 0
        for topic_handler in topic_handlers:
            num_channels += len(topic_handler.channel_dict)
        assert num_channels == 6
        data_client.mock_raw_data = np.random.random(num_channels)
        mock_raw_data_dict = {
            channel_name: value
            for channel_name, value in zip(
                data_client.channel_names, data_client.mock_raw_data
            )
        }

        await data_client.start()
        assert data_client.handle is not None
        assert not data_client.run_task.done()

        # Wait for data to be written, then check it.
        data_client.wrote_event.clear()
        await asyncio.wait_for(data_client.wrote_event.wait(), timeout=TIMEOUT)

        # Some fields of 3 topics should have been written.
        data = topic_handlers[0].topic.data_dict["labjack_test_1"]

        self.check_topic_handler(
            topic_handler=topic_handlers[0],
            topic_name="tel_temperature",
            sensor_name="labjack_test_1",
            field_name="temperature",
            location="somewhere, nowhere, somewhere else, guess",
            offset=1.5,
            scale=-2.1,
            channel_dict={0: "AIN0", 2: "AIN3", 3: "AIN2"},
            array_len=16,
        )
        self.check_topic_handler(
            topic_name="tel_pressure",
            topic_handler=topic_handlers[1],
            sensor_name="labjack_test_2",
            field_name="pressure",
            location="top of stack, bottom of stack",
            offset=0,
            scale=1,
            channel_dict={0: "AIN4", 1: "AIN5"},
            array_len=8,
        )
        self.check_topic_handler(
            topic_name="tel_temperature",
            topic_handler=topic_handlers[2],
            sensor_name="labjack_test_3",
            field_name="temperature",
            location="none, here",
            offset=0,
            scale=1,
            channel_dict={1: "AIN6"},
            array_len=16,
        )

        self.check_data(
            data=data,
            topic_handler=topic_handlers[0],
            raw_data_dict=mock_raw_data_dict,
        )
        data = topic_handlers[1].topic.data_dict["labjack_test_2"]
        self.check_data(
            data=data,
            topic_handler=topic_handlers[1],
            raw_data_dict=mock_raw_data_dict,
        )
        data = topic_handlers[2].topic.data_dict["labjack_test_3"]
        self.check_data(
            data=data,
            topic_handler=topic_handlers[2],
            raw_data_dict=mock_raw_data_dict,
        )

        await data_client.stop()
        assert data_client.handle is None
        assert data_client.run_task.done()

    async def test_registry(self) -> None:
        data_client_class = common.get_data_client_class("LabJackDataClient")
        assert data_client_class is labjack.LabJackDataClient

    def check_data(
        self,
        data: Any,
        topic_handler: labjack.TopicHandler,
        raw_data_dict: dict[str, float],
    ) -> None:
        """Check topic data.

        Parameters
        ----------
        data : Any
            The data
        topic_handler : labjack.TopicHandler
            Topic handler for this topic.
        sensor_name : str
            Sensor name.
        location: str
            Location string.
        field_name: str
            Name of array-valued field.
        raw_data_dict : `list` [`float`]
            Expected raw values, as a dict of channel_name: value
        """
        assert data.sensorName == topic_handler.sensor_name
        assert data.location == topic_handler.location
        data_arr = getattr(data, topic_handler.field_name)
        assert len(data_arr) == topic_handler.array_len
        for i in range(topic_handler.array_len):
            channel_name = topic_handler.channel_dict.get(i, None)
            if channel_name is None:
                assert math.isnan(data_arr[i])
            else:
                expected_scaled = (
                    raw_data_dict[channel_name] - topic_handler.offset
                ) * topic_handler.scale
                assert data_arr[i] == pytest.approx(expected_scaled)
                assert math.isfinite(data_arr[i])

    def check_topic_handler(
        self,
        topic_handler: labjack.TopicHandler,
        topic_name: str,
        sensor_name: str,
        field_name: str,
        location: str,
        offset: float,
        scale: float,
        array_len: int,
        channel_dict: dict[int, str],
    ) -> None:
        """Check the attributes of a topic handler.

        Parameters
        ----------
        topic_handler : labjack.TopicHandler
            Topic handler for this topic.
        topic_name : `str`
            Topic attribute name.
        sensor_name : `str`
            Expected sensor name.
        location: str
            Expected location string.
        field_name: `str`
            Expected name of array-valued field.
        array_len : `int`
            Expected length of the array.
        channel_dict : `dict` [`int`, `str`]
            Expected dict of array index: LabJack channel name
        """
        assert topic_handler.sensor_name == sensor_name
        assert topic_handler.topic.attr_name == topic_name
        assert topic_handler.location == location
        assert topic_handler.field_name == field_name
        assert topic_handler.offset == pytest.approx(offset)
        assert topic_handler.scale == pytest.approx(scale)
        assert topic_handler.array_len == array_len
        assert topic_handler.channel_dict == channel_dict

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
