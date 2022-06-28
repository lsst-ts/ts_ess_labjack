# This file is part of ts_ess_common.
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
        """Construct with good_full.yaml

        and compare values to that file.
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
        """Construct with good_minimal.yaml

        and compare values to that file.
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

        await data_client.start()
        assert data_client.handle is not None
        assert not data_client.run_task.done()

        # Wait for data to be written, then check it for plausibility
        # (we do not know what values will be written).
        data_client.wrote_event.clear()
        await asyncio.wait_for(data_client.wrote_event.wait(), timeout=TIMEOUT)

        # Some fields of 3 topics should have been written.
        data = topic_handlers[0].topic.data_dict["labjack_test_1"]
        self.check_data(
            data=data,
            sensor_name=data.sensorName,
            field_name="temperature",
            location="somewhere, nowhere, somewhere else, guess",
            good_indices={0, 2, 3},
            expected_len=16,
        )
        data = topic_handlers[1].topic.data_dict["labjack_test_2"]
        self.check_data(
            data=data,
            sensor_name=data.sensorName,
            field_name="pressure",
            location="top of stack, bottom of stack",
            good_indices={0, 1},
            expected_len=8,
        )
        data = topic_handlers[2].topic.data_dict["labjack_test_3"]
        self.check_data(
            data=data,
            sensor_name=data.sensorName,
            field_name="temperature",
            location="none, here",
            good_indices={1},
            expected_len=16,
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
        sensor_name: str,
        field_name: str,
        location: str,
        good_indices: set[int],
        expected_len: int,
    ) -> None:
        """Check topic data.

        We have no idea what the array values should be so just check
        that the expected indices are finite or nan.

        Parameters
        ----------
        data : Any
            The data
        sensor_name : str
            Sensor name.
        location: str
            Location string.
        field_name: str
            Name of array-valued field.
        good_indices : `set` [`int`]
            Indices of values expected to be finite.
        expected_len : `int`
            Expected length of the array.
        """
        assert data.sensorName == sensor_name
        assert data.location == location
        data_arr = getattr(data, field_name)
        assert len(data_arr) == expected_len
        for i in range(expected_len):
            if i in good_indices:
                assert math.isfinite(data_arr[i])
            else:
                assert math.isnan(data_arr[i])

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
