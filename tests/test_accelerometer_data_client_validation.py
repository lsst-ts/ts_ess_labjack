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

import copy
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

    async def test_good_full_two_accelerometers(self) -> None:
        config = self.get_and_validate_config("good_full_two_accelerometers.yaml")

        # Check against the values in file good_full.yaml.
        assert config.device_type == "T4"
        assert config.connection_type == "USB"
        assert config.max_frequency == 800
        assert config.accelerometers == [
            dict(
                sensor_name="alpha",
                location="upstairs",
                analog_inputs=[0, 2, 5],
                offsets=[0, -0.1, 1],
                scales=[1, 0.1, 2],
            ),
            dict(
                sensor_name="beta",
                location="downstairs",
                analog_inputs=[6, 1, 3],
                offsets=[0.1, 0.2, 0.3],
                scales=[0.4, 0.5, 0.6],
            ),
        ]

    async def test_good_minimal_one_accelerometer(self) -> None:
        config = self.get_and_validate_config("good_minimal_one_accelerometer.yaml")

        # Check against the values in file good_minimal.yaml.
        assert config.device_type == "T7"
        assert config.connection_type == "TCP"
        assert config.max_frequency == 1000
        assert config.accelerometers == [
            dict(
                sensor_name="chaos",
                location="auxtel",
                analog_inputs=[6, 3, 1],
                offsets=[0, 0.1, -0.2],
                scales=[1, 1.1, 1.2],
            ),
        ]

    async def test_wrong_fields(self) -> None:
        # Test missing a required field (that has no default).
        config = self.get_and_validate_config("good_minimal_one_accelerometer.yaml")
        good_config_dict = vars(config)
        for field_with_no_default in ("identifier", "max_frequency", "accelerometers"):
            bad_config_dict = good_config_dict.copy()
            del bad_config_dict[field_with_no_default]
            with pytest.raises(jsonschema.ValidationError):
                self.validator.validate(bad_config_dict)

        # Test adding a field
        good_config_dict = vars(config)
        bad_config_dict = good_config_dict.copy()
        bad_config_dict["no_such_field"] = 0
        with pytest.raises(jsonschema.ValidationError):
            self.validator.validate(bad_config_dict)

    async def test_wrong_accelerometer_fields(self) -> None:
        # Test missing a required field (that has no default).
        config = self.get_and_validate_config("good_minimal_one_accelerometer.yaml")
        good_config_dict = vars(config)
        for field in config.accelerometers[0]:
            bad_config_dict = copy.deepcopy(good_config_dict)
            del bad_config_dict["accelerometers"][0][field]
            with pytest.raises(jsonschema.ValidationError):
                self.validator.validate(bad_config_dict)

        # Test adding a field
        good_config_dict = vars(config)
        bad_config_dict = good_config_dict.copy()
        bad_config_dict["accelerometers"][0]["no_such_field"] = 0
        with pytest.raises(jsonschema.ValidationError):
            self.validator.validate(bad_config_dict)

    async def test_bad_analog_inputs(self) -> None:
        config = self.get_and_validate_config("good_minimal_one_accelerometer.yaml")
        good_config_dict = vars(config)

        # Test the wrong number of analog inputs.
        # Pick a subset of inputs from a list of arbitrary valid inputs.
        arbitrary_valid_inputs = [2, 1, 0, 3, 5, 8, 9, 21]
        for bad_num_values in range(7):
            if bad_num_values == 3:
                continue
            bad_analog_inputs = arbitrary_valid_inputs[0:bad_num_values]
            bad_config_dict = copy.deepcopy(good_config_dict)
            bad_config_dict["accelerometers"][0]["analog_inputs"] = bad_analog_inputs
            with pytest.raises(jsonschema.ValidationError):
                self.validator.validate(bad_config_dict)

        # Test negative analog inputs.
        for bad_value, index in itertools.product((-1, -2, -10), (0, 1, 2)):
            bad_config_dict = good_config_dict.copy()
            bad_config_dict["accelerometers"][0]["analog_inputs"][index] = bad_value
            with pytest.raises(jsonschema.ValidationError):
                self.validator.validate(bad_config_dict)

    async def test_bad_offsets_or_scales(self) -> None:
        config = self.get_and_validate_config("good_minimal_one_accelerometer.yaml")
        good_config_dict = vars(config)

        # Test the wrong number of analog inputs.
        # Pick a subset of inputs from a list of arbitrary valid inputs.
        arbitrary_valid_inputs = [2, 1, 0, 3, 5, 8, 9, 21]
        for field in ("offsets", "scales"):
            for bad_num_values in range(7):
                if bad_num_values == 3:
                    continue
                bad_values = arbitrary_valid_inputs[0:bad_num_values]
                bad_config_dict = copy.deepcopy(good_config_dict)
                bad_config_dict["accelerometers"][0][field] = bad_values
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
