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

__all__ = ["TopicHandler"]

import math
from collections.abc import Sequence

from lsst.ts import salobj, utils


class TopicHandler:
    """Put data for an ESS array-valued telemetry topic.

    Parameters
    ----------
    topic : `salobj.topics.WriteTopic`
        ESS array-valued topic to write.
    sensor_name : `str`
        The sensor name. Each instance of this topic that any ESS CSC writes
        must have a unique sensor name.
        Used to set the``sensorName`` field of the topic.
    field_name : `str`
        The name of the array-valued field in the topic.
    location : `str`
        Location of sensors, as a comma-separated string
        with len(channel_names) elements.
        Used to set the ``location`` field of the topic.
    channel_names : `List` [`str`]
        The LabJack channels to read, with one entry
        per field array value that you wish to write.
        Use "" for any array values you wish to skip.
        The list may be shorter than the field array;
        all skipped and unspecified values will be set to ``nan``.
        See the example in the Notes section.
    offset : `float`
        SAL value = offset + scale * LabJack value
    scale : `float`
        SAL value = offset + scale * LabJack value

    Notes
    -----
    Here is an example that assumes the topic is the ``temperature``
    ESS telemetry topic. This topic has one array field ``temperatures``
    with 16 elements.

    These inputs:

    * sensor_name = "topic_handler_example"
    * field_name = "temperatureItem"  # the only array field in this topic
    * location = "north, not used, west"
    * channel_names = ["AIN2", None, "AIN0"]

    will write these fields to the topic:

    * location = "north, not used, west"
    * numElements = 3
    * telemetry = [ain2, nan, ain0, nan, nan, ...]  # 16 values

    where:
    * ainX = offset + scale * raw_ainX
    * raw_ainX is the value read from LabJack channel AINX
    """

    def __init__(
        self,
        topic: salobj.topics.WriteTopic,
        sensor_name: str,
        location: str,
        field_name: str,
        channel_names: Sequence[str],
        offset: float = 0,
        scale: float = 1,
    ):
        self.topic = topic
        self.sensor_name = sensor_name
        self.location = location
        self.field_name = field_name
        self.offset = offset
        self.scale = scale

        topic_data = self.topic.DataType()
        if not hasattr(topic_data, field_name):
            raise ValueError(
                f"Cannot find field_name={field_name!r} in topic {topic.attr_name}."
            )
        missing_fields = [
            required_field
            for required_field in ("sensorName", "timestamp", "numChannels", "location")
            if not hasattr(topic_data, required_field)
        ]
        if missing_fields:
            raise ValueError(
                f"Cannot find required field(s) {missing_fields} in topic {topic.attr_name}. "
                "These should be present for all ESS array-valued telemetry topics."
            )

        field_data = getattr(topic_data, field_name)
        if not isinstance(field_data, list):
            raise ValueError(f"Field {topic.attr_name}.{field_name} is not array")
        self.array_len = len(field_data)
        # dict of array index: LabJack channel name.
        self.channel_dict = {i: name for i, name in enumerate(channel_names) if name}
        bad_indices = [i for i in self.channel_dict if i >= self.array_len]
        if bad_indices:
            raise ValueError(
                f"Field {topic.attr_name}.{field_name} only has {len(field_data)} elements; "
                f"one or more indices {bad_indices} is too big."
            )
        self.num_channels = max(i for i in self.channel_dict) + 1

    async def write_telemetry(self, telemetry_dict: dict[str, float]) -> None:
        """Write telemetry to the topic.

        Parameters
        ----------
        telemetry_dict : `dict` [`str`, `float`]
            dict of LabJack channel name: raw value.
            This may include more channels than this topic needs.
            SAL value = (LabJack value - offset) * scale
        """
        telemetry_arr = [math.nan] * self.array_len
        for i, channel_name in self.channel_dict.items():
            telemetry_arr[i] = (telemetry_dict[channel_name] - self.offset) * self.scale
        telemetry_kwarg = {self.field_name: telemetry_arr}
        await self.topic.set_write(
            sensorName=self.sensor_name,
            timestamp=utils.current_tai(),
            location=self.location,
            numChannels=self.num_channels,
            **telemetry_kwarg,
        )
