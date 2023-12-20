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

# Standard timeout in seconds
TIMEOUT = 5


class DataClientTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.data_dir = (
            pathlib.Path(__file__).parent / "data" / "config" / "data_client"
        )
        self.config_schema = labjack.LabJackDataClient.get_config_schema()
        self.validator = salobj.DefaultingValidator(self.config_schema)

    async def test_good_full(self) -> None:
        config = self.get_and_validate_config("good_full.yaml")

        # Check against the values in file good_full.yaml.
        assert config.device_type == "T4"
        assert config.connection_type == "USB"
        assert config.poll_interval == pytest.approx(0.2)
        assert config.processor == "AuxTelCameraCoolantPressureProcessor"
        assert len(config.channel_names) == len(config.offsets)
        assert len(config.channel_names) == len(config.scales)

    async def test_good_minimal(self) -> None:
        config = self.get_and_validate_config("good_minimal.yaml")

        # Check against the values in file good_minimal.yaml.
        assert config.device_type == "T7"
        assert config.connection_type == "TCP"
        assert config.poll_interval == 1
        assert config.processor == "AuxTelCameraCoolantPressureProcessor"
        assert len(config.channel_names) == len(config.offsets)
        assert len(config.channel_names) == len(config.scales)

    async def test_bad(self) -> None:
        for path in self.data_dir.glob("bad_*.yaml"):
            config_dict = self.get_config_dict(path)
            with pytest.raises(jsonschema.ValidationError):
                self.validator.validate(config_dict)

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
