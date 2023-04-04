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

__all__ = [
    "MockEssArrayTopic",
    "MockESSAccelerometerPSDTopic",
    "MockESSAccelerometerTopic",
]

import collections
import copy
import dataclasses
from collections.abc import Sequence
from typing import Any, Type


class GetListOfZeros:
    """Functor to return a list of zeros of a specified length.

    Intended as a default_factory for array-valued fields in a dataclass.
    """

    def __init__(self, field_len: int) -> None:
        self.field_len = field_len

    def __call__(self) -> list[float]:
        return [0] * self.field_len


class BaseMockEssTopic:
    """Base class for mock ESS telemetry topics.

    Parameters
    ----------
    attr_name : `str`
        Topic attribute name, e.g. "tel_temperature".
    DataType : `dataclasses.dataclass`
        Topic data class.

    Attributes
    ----------
    DataType : Type[`dataclasses.dataclass`]
        Dataclass class (not instance) for topic data.
    data : `dataclasses.dataclass`
        The data most recently written by `set`
    data_dict : `dict` [`str`, `list` [dataclass]]
        The data written by `set_write`,
        as a dict of sensor_name: list of data written.
    """

    def __init__(self, attr_name: str, DataType: Type[Any]) -> None:
        self.attr_name = attr_name
        self.DataType = DataType
        self.data = DataType()
        # Dict of sensor name: list of data written for that sensor
        self.data_dict: dict[str, Any] = collections.defaultdict(list)

        # Dict of field_name: array length for array-valued fields.
        self._array_field_lengths = {
            field: len(value)
            for field, value in vars(self.data).items()
            if isinstance(value, list)
        }
        self._field_names = vars(self.data).keys()

    def set(self, **kwargs: Any) -> Any:
        """Set self.data.

        Parameters
        ----------
        sensorName : `str`
            Sensor name.
        timestamp : `float`
            Time at which data was measured, as TAI unix.
        numChannels : `int`
            The number of valid channels.
            Must be >= 0 and < field_len.

        Returns
        -------
        data : `dataclasses.dataclass`
            A copy of the data written.
        """
        invalid_keys = kwargs.keys() - self._field_names
        if invalid_keys:
            raise ValueError(
                f"Unrecognized field names {sorted(invalid_keys)} in {kwargs}"
            )

        for field_name, new_value in kwargs.items():
            field_len = self._array_field_lengths.get(field_name)
            if field_len is not None:
                old_value = getattr(self.data, field_name)
                new_len = len(new_value)
                if field_len != new_len:
                    raise ValueError(
                        f"{field_name} has length {field_len} != {new_len}=len(new value)"
                    )
                new_value = list(new_value) + old_value[new_len:]
            setattr(self.data, field_name, new_value)
        return copy.deepcopy(self.data)

    async def set_write(self, **kwargs: Any) -> None:
        """Set self.data and cache a deep copy in self.data_dict."""
        new_data = self.set(**kwargs)
        self.data_dict[self.data.sensorName].append(new_data)

    async def write(self) -> None:
        pass


class MockEssArrayTopic(BaseMockEssTopic):
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
        The data most recently written by `set_write`
        as a dict of sensor_name: data.
    """

    def __init__(self, attr_name: str, field_name: str, field_len: int) -> None:
        if field_len <= 0:
            raise ValueError(f"field_len={field_len} must be positive.")
        self.field_name = field_name
        self.field_len = field_len

        get_zeros = GetListOfZeros(field_len)

        DataType = dataclasses.make_dataclass(
            cls_name="DataType",
            fields=[
                ("sensorName", str, dataclasses.field(default="0")),  # type: ignore
                ("timestamp", float, dataclasses.field(default=0)),  # type: ignore
                ("numChannels", int, dataclasses.field(default=0)),  # type: ignore
                (
                    self.field_name,
                    Sequence[float],
                    dataclasses.field(default_factory=get_zeros),  # type: ignore
                ),
                ("location", str, dataclasses.field(default="")),  # type: ignore
            ],
        )

        super().__init__(attr_name=attr_name, DataType=DataType)


class MockESSAccelerometerPSDTopic(BaseMockEssTopic):
    """Mock ESS accelerometerPSD topic.

    Attributes
    ----------
    data : `dataclasses.dataclass`
        The data most recently written by `set`
    data_dict : `dict` [`str`, dataclass]
        The data most recently written by `set_write`
        as a dict of sensor_name: data.
    """

    def __init__(self) -> None:
        get_zeros = GetListOfZeros(field_len=201)

        DataType = dataclasses.make_dataclass(
            cls_name="DataType",
            fields=[
                ("sensorName", str, dataclasses.field(default="")),  # type: ignore
                ("timestamp", float, dataclasses.field(default=0)),  # type: ignore
                ("maxPSDFrequency", float, dataclasses.field(default=0)),  # type: ignore
            ]
            + [
                (
                    f"accelerationPSD{axis}",
                    list[float],
                    dataclasses.field(default_factory=get_zeros),  # type: ignore
                )
                for axis in ("X", "Y", "Z")
            ]
            + [
                ("location", str, dataclasses.field(default="")),  # type: ignore
            ],
        )

        super().__init__(attr_name="tel_accelerometerPSD", DataType=DataType)


class MockESSAccelerometerTopic(BaseMockEssTopic):
    """Mock ESS accelerometer topic.

    Attributes
    ----------
    data : `dataclasses.dataclass`
        The data most recently written by `set`
    data_dict : `dict` [`str`, dataclass]
        The data most recently written by `set_write`
        as a dict of sensor_name: data.
    """

    def __init__(self) -> None:
        get_zeros = GetListOfZeros(field_len=400)

        DataType = dataclasses.make_dataclass(
            cls_name="DataType",
            fields=[
                ("sensorName", str, dataclasses.field(default="")),  # type: ignore
                ("timestamp", float, dataclasses.field(default=0)),  # type: ignore
                ("interval", float, dataclasses.field(default=0)),  # type: ignore
            ]
            + [
                (
                    f"acceleration{axis}",
                    list[float],
                    dataclasses.field(default_factory=get_zeros),  # type: ignore
                )
                for axis in ("X", "Y", "Z")
            ]
            + [
                ("location", str, dataclasses.field(default="")),  # type: ignore
            ],
        )

        super().__init__(attr_name="tel_accelerometer", DataType=DataType)
