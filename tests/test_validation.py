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

import logging
import pathlib
import types
import unittest
from typing import Any, Dict, Union

import jsonschema
import pytest
import yaml

from lsst.ts import salobj
from lsst.ts.ess import labjack

PathT = Union[str, pathlib.Path]

logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s", level=logging.DEBUG
)

TIMEOUT = 5
"""Standard timeout in seconds."""


class DataClientTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.data_dir = pathlib.Path(__file__).parent / "data"
        self.config_schema = labjack.LabJackDataClient.get_config_schema()
        self.validator = salobj.DefaultingValidator(self.config_schema)

    def test_good_full(self) -> None:
        config = self.get_and_validate_config("good_full.yaml")

        # Check against the values in file good_full.yaml
        assert config.device_type == "T4"
        assert config.connection_type == "USB"
        assert config.poll_interval == 0.2
        assert len(config.topics) == 3
        topics = [types.SimpleNamespace(**topic_dict) for topic_dict in config.topics]
        assert topics[0].topic_name == "tel_temperature"
        assert topics[0].sensor_name == "labjack_test_1"
        assert topics[0].field_name == "temperature"
        assert topics[0].location == "somewhere, nowhere, somewhere else, guess"
        assert topics[0].channel_names == ["AIN0", "", "AIN3", "AIN2"]
        assert topics[0].offset == 1.5
        assert topics[0].scale == -2.1

        assert topics[1].topic_name == "tel_pressure"
        assert topics[1].sensor_name == "labjack_test_2"
        assert topics[1].field_name == "pressure"
        assert topics[1].location == "top of stack, bottom of stack"
        assert topics[1].channel_names == ["AIN4", "AIN5"]
        assert topics[1].offset == 0
        assert topics[1].scale == 1

        assert topics[2].topic_name == "tel_temperature"
        assert topics[2].sensor_name == "labjack_test_3"
        assert topics[2].field_name == "temperature"
        assert topics[2].location == "none, here"
        assert topics[2].channel_names == ["", "AIN6"]
        assert topics[2].offset == 0
        assert topics[2].scale == 1

    def test_good_minimal(self) -> None:
        config = self.get_and_validate_config("good_minimal.yaml")

        # Check against the values in file good_minimal.yaml
        assert config.device_type == "T7"
        assert config.connection_type == "TCP"
        assert config.poll_interval == 1
        assert len(config.topics) == 1
        topics = [types.SimpleNamespace(**topic_dict) for topic_dict in config.topics]
        assert topics[0].topic_name == "tel_temperature"
        assert topics[0].sensor_name == "labjack_test_1"
        assert topics[0].field_name == "temperature"
        assert topics[0].location == "none, here"
        assert topics[0].channel_names == ["", "AIN2"]
        assert topics[0].offset == 0
        assert topics[0].scale == 1

    def test_bad(self) -> None:
        for path in self.data_dir.glob("bad_*.yaml"):
            config_dict = self.get_config_dict(path)
            with pytest.raises(jsonschema.ValidationError):
                self.validator.validate(config_dict)

    def get_and_validate_config(self, filename: PathT) -> types.SimpleNamespace:
        raw_config_dict = self.get_config_dict(filename)
        config_dict = self.validator.validate(raw_config_dict)
        return types.SimpleNamespace(**config_dict)

    def get_config_dict(self, filename: PathT) -> Dict[str, Any]:
        """Get a config dict from tests/data.

        Parameters
        ----------
        filename : `str` or `pathlib.Path`
            Name of config file, including ".yaml" suffix.

        Returns
        -------
        config_dict : `dict` [`str`, Any]
            The config dict.
        """
        with open(self.data_dir / filename, "r") as f:
            return yaml.safe_load(f.read())
