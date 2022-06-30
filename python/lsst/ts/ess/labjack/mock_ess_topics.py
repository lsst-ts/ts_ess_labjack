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

__all__ = ["MockEssArrayTopic", "MockESSAccelerometerPSDTopic"]

import dataclasses
from collections.abc import Sequence
from typing import Any


class MockEssArrayTopic:
    """Mock array-valued ESS telemetry topic.

    Parameters
    ----------
    attr_name : `str`
        Topic attribute name, e.g. "tel_temperature".
    field_name : `str`
        The name of the array-valued field.
    field_len : `int`
        The length of the array-valued field.

    Attributes
    ----------
    data_dict : `dict` [`str`, dataclass]
        The data most recently written by `set_put`
        as a dict of sensor_name: data.
    """

    def __init__(self, attr_name: str, field_name: str, field_len: int) -> None:
        if field_len <= 0:
            raise ValueError(f"field_len={field_len} must be positive.")
        self.attr_name = attr_name
        self.field_name = field_name
        self.field_len = field_len

        def get_zeros() -> list[float]:
            """Get a list of field_len zeros.

            Needed in order to make the default for the array field
            a mutable type (a tuple isn't good enough because some
            code really cares that it is a list).
            """
            return [0] * field_len

        self.DataType = dataclasses.make_dataclass(
            cls_name="DataType",
            fields=[
                ("sensorName", str, dataclasses.field(default="0")),  # type: ignore
                ("timestamp", float, dataclasses.field(default=0)),  # type: ignore
                ("numChannels", int, dataclasses.field(default=0)),  # type: ignore
                (
                    field_name,
                    Sequence[float],
                    dataclasses.field(default_factory=get_zeros),  # type: ignore
                ),
                ("location", str, dataclasses.field(default="")),  # type: ignore
            ],
        )
        self.data_dict: dict[str, Any] = dict()

    async def set_write(
        self,
        sensorName: str,
        timestamp: float,
        numChannels: int,
        location: str,
        **kwargs: Sequence[float],
    ) -> None:
        """Set self.data_dict[sensorName] = data.

        where data is a dataclass that mimics the SAL topic data type,
        but without the private fields or the ESSID field.
        The data also has no get_vars method.

        Parameters
        ----------
        sensorName : `str`
            Sensor name.
        timestamp : `float`
            Time at which data was measured, as TAI unix.
        numChannels : `int`
            The number of valid channels.
            Must be >= 0 and < field_len.
        """
        if numChannels < 0 or numChannels >= self.field_len:
            raise ValueError(f"numChannels={numChannels} <0 or >= {self.field_len}]")
        array = kwargs.get(self.field_name, None)
        if len(kwargs) != 1 or array is None:
            raise ValueError(
                f"kwargs={kwargs} must have exactly one item, with key {self.field_name}"
            )
        if len(array) != self.field_len:
            raise ValueError(
                f"{self.field_name}={array} must have {self.field_len} elements"
            )
        self.data_dict[sensorName] = self.DataType(
            sensorName=sensorName,
            timestamp=timestamp,
            numChannels=numChannels,
            location=location,
            **kwargs,
        )


class MockESSAccelerometerPSDTopic:
    """Mock ESS accelerometerPSD topic.

    Attributes
    ----------
    data : `dataclasses.dataclass`
        The data most recently written by `set`
    """

    def __init__(self) -> None:
        self.attr_name = "tel_accelerometerPSD"

        def get_zeros() -> list[float]:
            """Get a list of 200 zeros.

            Needed in order to make the default for the array field
            a mutable type (a tuple isn't good enough because some
            code really cares that it is a list).
            """
            return [0] * 200

        self.DataType = dataclasses.make_dataclass(
            cls_name="DataType",
            fields=[
                ("sensorName", str, dataclasses.field(default="")),  # type: ignore
                ("timestamp", float, dataclasses.field(default=0)),  # type: ignore
                ("interval", float, dataclasses.field(default=0)),  # type: ignore
                ("minPSDFrequency", float, dataclasses.field(default=0)),  # type: ignore
                ("maxPSDFrequency", float, dataclasses.field(default=0)),  # type: ignore
                ("numDataPoints", int, dataclasses.field(default=0)),  # type: ignore
                (
                    "accelerationPSDX",
                    list[float],
                    dataclasses.field(default_factory=get_zeros),  # type: ignore
                ),
                (
                    "accelerationPSDY",
                    list[float],
                    dataclasses.field(default_factory=get_zeros),  # type: ignore
                ),
                (
                    "accelerationPSDZ",
                    list[float],
                    dataclasses.field(default_factory=get_zeros),  # type: ignore
                ),
                ("location", str, dataclasses.field(default="")),  # type: ignore
            ],
        )
        self.data = self.DataType()

    def set(self, **kwargs: Any) -> None:
        """Set self.data_dict[sensorName] = data.

        where data is a dataclass that mimics the SAL topic data type,
        but without the private fields or the ESSID field.
        The data also has no get_vars method.

        Parameters
        ----------
        sensorName : `str`
            Sensor name.
        timestamp : `float`
            Time at which data was measured, as TAI unix.
        numChannels : `int`
            The number of valid channels.
            Must be >= 0 and < field_len.
        """
        for axis in ("X", "Y", "Z"):
            field_name = f"accelerationPSD{axis}"
            array_data = kwargs.pop(field_name, None)
            if array_data is None:
                continue

            array_len = len(array_data)
            if len(array_data) > 200:
                raise ValueError(
                    f"{field_name}={array_data} must have no more than 200 elements"
                )
            old_data = getattr(self.data, field_name)
            full_data = list(array_data[0:array_len]) + list(old_data[array_len:200])
            setattr(self.data, field_name, full_data)

        for field_name, value in kwargs.items():
            setattr(self.data, field_name, value)

    async def set_write(self, **kwargs: Any) -> None:
        self.set(**kwargs)
