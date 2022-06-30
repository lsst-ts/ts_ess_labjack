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

import itertools
import logging
import pathlib
import types
import unittest
from typing import Any, TypeAlias

import jsonschema
import pytest
import yaml

from lsst.ts import salobj
from lsst.ts.ess import labjack

PathT: TypeAlias = str | pathlib.Path

logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s", level=logging.DEBUG
)

# Standard timeout in seconds.
TIMEOUT = 5


class AccelerationDataClientTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.data_dir = (
            pathlib.Path(__file__).parent
            / "data"
            / "config"
            / "accelerometer_data_client"
        )
        self.config_schema = labjack.LabJackAccelerometerDataClient.get_config_schema()
        self.validator = salobj.DefaultingValidator(self.config_schema)

    async def test_good_full(self) -> None:
        config = self.get_and_validate_config("good_full.yaml")

        # Check against the values in file good_full.yaml.
        assert config.device_type == "T4"
        assert config.connection_type == "USB"
        assert config.sensor_name == "alt_accel"
        assert config.location == "somewhere"
        assert config.min_frequency == 20
        assert config.max_frequency == 500
        assert config.num_frequencies == 97
        assert config.analog_inputs == [0, 2, 5]
        assert config.scale == 10

    async def test_good_minimal(self) -> None:
        config = self.get_and_validate_config("good_minimal.yaml")

        # Check against the values in file good_minimal.yaml.
        assert config.device_type == "T7"
        assert config.connection_type == "TCP"
        assert config.sensor_name == "accel"
        assert config.location == "auxtel"
        assert config.min_frequency == 0
        assert config.max_frequency == 1000
        assert config.num_frequencies == 200
        assert config.analog_inputs == [6, 3, 1]
        assert config.scale == 1

    async def test_missing_field(self) -> None:
        config = self.get_and_validate_config("good_minimal.yaml")
        fields_with_defaults = {
            "device_type",
            "connection_type",
            "min_frequency",
            "num_frequencies",
        }
        good_config_dict = vars(config)
        for missing_field in good_config_dict:
            if missing_field in fields_with_defaults:
                continue
            bad_config_dict = good_config_dict.copy()
            del bad_config_dict[missing_field]
            with pytest.raises(jsonschema.ValidationError):
                self.validator.validate(bad_config_dict)

    async def test_bad_analog_inputs(self) -> None:
        config = self.get_and_validate_config("good_minimal.yaml")
        good_config_dict = vars(config)

        # Test the wrong number of analog inputs.
        # Pick a subset of inputs from a list of arbitrary valid inputs.
        arbitrary_valid_inputs = [2, 1, 0, 3, 5, 8, 9, 21]
        for bad_num_analog_inputs in range(7):
            if bad_num_analog_inputs == 3:
                continue
            bad_analog_inputs = arbitrary_valid_inputs[0:bad_num_analog_inputs]
            bad_config_dict = good_config_dict.copy()
            bad_config_dict["analog_inputs"] = bad_analog_inputs
            with pytest.raises(jsonschema.ValidationError):
                self.validator.validate(bad_config_dict)

        # Test negative analog inputs.
        for bad_value, index in itertools.product((-1, -2, -10), (0, 1, 2)):
            bad_config_dict = good_config_dict.copy()
            bad_config_dict["analog_inputs"][index] = bad_value
            print("bad_config_dict=", bad_config_dict)
            with pytest.raises(jsonschema.ValidationError):
                self.validator.validate(bad_config_dict)

    def get_and_validate_config(self, filename: PathT) -> types.SimpleNamespace:
        raw_config_dict = self.get_config_dict(filename)
        config_dict = self.validator.validate(raw_config_dict)
        return types.SimpleNamespace(**config_dict)

    def get_config_dict(self, filename: PathT) -> dict[str, Any]:
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
