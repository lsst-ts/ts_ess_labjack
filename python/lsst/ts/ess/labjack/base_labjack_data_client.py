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

__all__ = ["BaseLabJackDataClient"]

import abc
import asyncio
import concurrent
import logging
import socket
import types
from collections.abc import Callable
from typing import Any

# Hide mypy error `Module "labjack" has no attribute "ljm"`.
from labjack import ljm  # type: ignore
from lsst.ts import salobj
from lsst.ts.ess import common

# Time limit for connecting to the LabJack (seconds).
CONNECT_TIMEOUT = 5

# LabJack's special identifier to run in simulation mode.
MOCK_IDENTIFIER = "LJM_DEMO_MODE"


class BaseLabJackDataClient(common.data_client.BaseDataClient, abc.ABC):
    """Base class for ESS data clients that read a LabJack T7 or similar.

    Parameters
    ----------
    name : `str`
    config : `types.SimpleNamespace`
        The configuration, after validation by the schema returned
        by `get_config_schema` and conversion to a types.SimpleNamespace.
    topics : `salobj.Controller`
        The telemetry topics this model can write, as a struct with attributes
        such as ``tel_temperature``.
    log : `logging.Logger`
        Logger.
    simulation_mode : `int`, optional
        Simulation mode; 0 for normal operation.

    Notes
    -----
    In simulation mode the mock LabJack returns unspecified values,
    and those values may change in future versions of the LabJack software.
    """

    def __init__(
        self,
        config: types.SimpleNamespace,
        topics: salobj.Controller | types.SimpleNamespace,
        log: logging.Logger,
        simulation_mode: int = 0,
    ) -> None:
        # handle to LabJack device.
        self.handle = None

        # The thread pool executor used by `run_in_thread`.
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        super().__init__(
            config=config, topics=topics, log=log, simulation_mode=simulation_mode
        )

    @property
    def connected(self) -> bool:
        return self.handle is not None

    def descr(self) -> str:
        return f"identifier={self.config.identifier}"

    async def run_in_thread(self, func: Callable[[], Any], timeout: float) -> Any:
        """Run a blocking function in a thread pool executor.

        Only one function will run at a time, because all calls use the same
        thread pool executor, which only has a single thread.

        Parameters
        ----------
        func : `Callable`
            The blocking function to run;
            The function must take no arguments.
            For example: ``self._blocking_connect``.
        timeout : `float`
            Time limit (seconds).
        """
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(self._thread_pool, func), timeout=timeout
            )
        except asyncio.CancelledError:
            self.log.info(
                f"run_in_thread cancelled while running blocking function {func}."
            )
        except Exception:
            self.log.exception(f"Blocking function {func} failed.")
            raise

    async def connect(self) -> None:
        """Connect to the LabJack.

        Disconnect first, if connected.

        Raises
        ------
        RuntimeError
            If we cannot read the data.
        """
        self.log.info(
            f"Connect to LabJack {self.config.device_type}, "
            f"config.identifier={self.config.identifier!r}, "
            f"config.connection_type={self.config.connection_type}"
        )
        await self.run_in_thread(func=self._blocking_connect, timeout=CONNECT_TIMEOUT)

    async def disconnect(self) -> None:
        """Disconnect from the LabJack. A no-op if disconnected."""
        try:
            await self.run_in_thread(
                func=self._blocking_disconnect, timeout=CONNECT_TIMEOUT
            )
        finally:
            self.handle = None

    def _blocking_connect(self) -> None:
        """Connect to the LabJack and stop streaming (if active).

        Disconnect first, if connected.

        Call in a thread to avoid blocking the event loop.
        """
        if self.handle is not None:
            self.log.warning("Already connected; disconnecting and reconnecting")
            self._blocking_disconnect()

        if self.simulation_mode == 0:
            identifier = self.config.identifier
            if self.config.connection_type in {"TCP", "WIFI"}:
                # Resolve domain name, since ljm does not do this.
                identifier = socket.gethostbyname(identifier)
                self.log.info(f"resolved identifier={identifier!r}")
        else:
            identifier = MOCK_IDENTIFIER
            self.log.info(f"simulation mode, so identifier changed to {identifier!r}")

        self.handle = ljm.openS(
            self.config.device_type, self.config.connection_type, identifier
        )
        self._blocking_stop_data_stream()

    def _blocking_disconnect(self) -> None:
        """Disconnect from the LabJack after stopping streaming.

        A no-op if disconnected.

        Call in a thread to avoid blocking the event loop.
        """
        if self.handle is not None:
            try:
                self._blocking_stop_data_stream()
                ljm.close(self.handle)
            finally:
                self.handle = None

    def _blocking_stop_data_stream(self) -> None:
        """Try to stop streaming data from the LabJack.

        This is intended as a "best effort" basis and is meant to always
        be safe to call. Thus this traps and logs ljm.LJMError (which is
        the only expected exception). We recommend calling this:

        * Just before disconnecting (self.disconnect does this),
          in case the data client has enabled streaming.
        * Just after connecting (self.connect does this),
          in case the LabJack was accidentally left in streaming mode.

        Call in a thread to avoid blocking the event loop.


        """
        # LabJack ljm demo mode does not support streaming,
        # but this call seems to work anyway.
        try:
            ljm.eStreamStop(self.handle)
        except ljm.LJMError as e:
            # Note: I would rather compare e.errorCode to
            # ljm.errorcodes.STREAM_NOT_RUNNING = 1303,
            # but the error code is 2620, which does not match any constant
            # in ljm.errorcodes.
            if e.errorString == "STREAM_NOT_RUNNING":
                # Deliberately ignore.
                pass
            else:
                self.log.warning(
                    "Could not stop LabJack streaming, but continuing anyway: "
                    f"{e!r}: {e.errorString=}"
                )
