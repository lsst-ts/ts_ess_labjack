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

import asyncio
import concurrent
import logging
import socket
import types
from collections.abc import Callable
from typing import Any

# Hide mypy error `Module "labjack" has no attribute "ljm"`
from labjack import ljm  # type: ignore

from lsst.ts.ess import common

from lsst.ts import salobj

# Time limit for connecting to the LabJack (seconds)
CONNECT_TIMEOUT = 5

# LabJack's special identifier to run in simulation mode.
MOCK_IDENTIFIER = "LJM_DEMO_MODE"


class BaseLabJackDataClient(common.BaseDataClient):
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
        # handle to LabJack device
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
        """Connect to the LabJack.

        Disconnect first, if connected.

        Call in a thread to avoid blocking the event loop.
        """
        if self.handle is not None:
            self.log.warning("Already connected; disconnecting and reconnecting")
            self._blocking_disconnect()

        if self.simulation_mode == 0:
            identifier = self.config.identifier
            if self.config.connection_type in {"TCP", "WIFI"}:
                # Resolve domain name, since ljm does not do this
                identifier = socket.gethostbyname(identifier)
                self.log.info(f"resolved identifier={identifier!r}")
        else:
            identifier = MOCK_IDENTIFIER
            self.log.info(f"simulation mode, so identifier changed to {identifier!r}")

        self.handle = ljm.openS(
            self.config.device_type, self.config.connection_type, identifier
        )

    def _blocking_disconnect(self) -> None:
        """Disconnect from the LabJack. A no-op if disconnected.

        Call in a thread to avoid blocking the event loop.
        """
        if self.handle is not None:
            try:
                ljm.close(self.handle)
            finally:
                self.handle = None