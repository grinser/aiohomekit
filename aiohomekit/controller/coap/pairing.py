#
# Copyright 2022 aiohomekit team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from datetime import timedelta
from typing import Any

from aiohomekit.controller.abstract import AbstractController, AbstractPairingData
from aiohomekit.exceptions import AccessoryDisconnectedError
from aiohomekit.model import Accessories, AccessoriesState, Transport
from aiohomekit.model.characteristics import CharacteristicPermissions
from aiohomekit.protocol.statuscodes import HapStatusCode
from aiohomekit.utils import async_create_task
from aiohomekit.uuid import normalize_uuid
from aiohomekit.zeroconf import HomeKitService, ZeroconfPairing

from .connection import PAIR_VERIFY_ATTEMPTS, CoAPHomeKitConnection

logger = logging.getLogger(__name__)


class CoAPPairing(ZeroconfPairing):
    def __init__(
        self,
        controller: AbstractController,
        pairing_data: AbstractPairingData,
        description: HomeKitService | None = None,
    ) -> None:
        self.connection = CoAPHomeKitConnection(
            self, pairing_data["AccessoryIP"], pairing_data["AccessoryPort"]
        )
        self.connection_future = None
        self.connection_lock = asyncio.Condition()
        self.pairing_data = pairing_data
        # Assigned directly, and before super().__init__, exactly as
        # BlePairing does it: AbstractPairing never touches self.description,
        # and routing it through _async_description_update instead would
        # schedule _process_config_changed -- the advertised config number
        # always exceeds our -1 -- so the pairing dialog would enumerate after
        # all, which is the thing having a description here avoids.
        self.description = description

        super().__init__(controller, pairing_data)

    def _async_endpoint_changed(self) -> None:
        """The IP/Port has changed, so close connection if active then reconnect."""
        self.connection.address = f"[{self.description.address}]:{self.description.port}"
        async_create_task(self.connection.reconnect_soon())

    @property
    def is_connected(self):
        return self.connection.is_connected

    @property
    def is_available(self) -> bool:
        """Returns true if the device is currently available."""
        return self.connection.is_connected

    @property
    def transport(self) -> Transport:
        """The transport used for the connection."""
        return Transport.COAP

    @property
    def name(self) -> str:
        """Return the name of the pairing with the address."""
        if self.description:
            return f"{self.description.name} [{self.connection.address}] (id={self.id})"
        return f"[{self.connection.address}] (id={self.id})"

    @property
    def poll_interval(self) -> timedelta:
        """Returns how often the device should be polled."""
        return timedelta(minutes=1)

    async def _ensure_connected(self, pair_verify_attempts: int = 1, enumerate_database: bool = True):
        """Connect if needed.

        `pair_verify_attempts` is the retry budget handed to the connection.
        It defaults to a single attempt: on an accessory that is not answering,
        each extra attempt holds connection_lock for another timeout, and every
        caller here is either a poll or a read that its own caller will retry.
        Only an interactive setup raises it -- see
        async_populate_accessories_state.

        `enumerate_database` is False for pairing operations, which need one
        characteristic and locate it themselves. Enumerating on their behalf is
        what makes an unpair miss the controller's deadline on an accessory
        that has to be enumerated by walking.
        """
        primary = False
        # let in one coroutine at a time
        async with self.connection_lock:
            if self._shutdown:
                return

            if not self.connection.is_connected:
                # if there isn't a connection in progress, we're in the driver's seat
                if self.connection_future is None:
                    primary = True
                    # The future covers establishing the session and nothing
                    # else. Enumeration happens after it, below: a caller that
                    # only needs a session -- an unpair, which the controller is
                    # timing -- must not wait out a ~300-request walk that
                    # somebody else's poll started. That wait is how an unpair
                    # gets abandoned, which orphans the pairing on the
                    # accessory. The race is routine rather than exotic:
                    # load_pairing schedules _process_config_changed in the
                    # background whenever zeroconf has a cached discovery, and
                    # the controller calls remove_pairing right afterwards.
                    self.connection_future = self.connection.connect(
                        self.pairing_data,
                        attempts=pair_verify_attempts,
                        enumerate_database=False,
                    )
                else:
                    # we'll wait on the primary coroutine & copy how it returns
                    # this drops the lock and reacquires it when we're notified
                    await self.connection_lock.wait()
                    # if the primary coroutine failed to connect, we also raise
                    if not self.connection.is_connected:
                        raise AccessoryDisconnectedError("primary coroutine failed to connect")

        if primary:
            try:
                # await the connection outside of the lock
                # this allows other coroutines to show up & wait
                await self.connection_future
            except BaseException:
                raise AccessoryDisconnectedError("failed to connect")
            else:
                # in case this was a reconnect, re-subscribe
                if len(self.subscriptions):
                    logger.debug(
                        "(Re-)subscribing to %d characteristics: %r"
                        % (len(self.subscriptions), self.subscriptions)
                    )
                    await self.connection.subscribe_to(list(self.subscriptions))
                self._callback_availability_changed(True)
            finally:
                # until we re-acquire the lock & clear connection_future,
                # other coroutines that show up will all hit the .wait() path.
                async with self.connection_lock:
                    # clear the flag indicating a connection is in progress
                    self.connection_future = None
                    # wake up any coroutines that showed up while we were connecting
                    self.connection_lock.notify_all()

        # Connected is not the same as ready: a session raised for a pairing
        # operation has no database behind it, and the next characteristic read
        # dereferences info unguarded. This check is racy by nature -- every
        # waiter released by the primary sees info as None at the same moment --
        # so the decision is re-taken under the enumeration lock, which is what
        # makes two callers arriving together cost one enumeration rather than
        # two.
        if enumerate_database and self.connection.info is None:
            await self.connection.get_accessory_info(
                verify_attempts=pair_verify_attempts, only_if_missing=True
            )

        return

    async def close(self) -> None:
        if self.connection.is_connected:
            await self.unsubscribe(list(self.subscriptions))

    def event_received(self, event):
        self._callback_listeners(event)

    async def identify(self):
        await self._ensure_connected()
        return await self.connection.do_identify()

    async def list_accessories_and_characteristics(
        self, pair_verify_attempts: int = 1
    ) -> list[dict[str, Any]]:
        # enumerate_database=False: this method does its own read below, and
        # letting _ensure_connected do one first would cost a second full pass
        # over every readable characteristic.
        await self._ensure_connected(pair_verify_attempts, enumerate_database=False)

        accessories = await self.connection.get_accessory_info(verify_attempts=pair_verify_attempts)

        for accessory in accessories:
            for service in accessory["services"]:
                service["type"] = normalize_uuid(service["type"])

                for characteristic in service["characteristics"]:
                    characteristic["type"] = normalize_uuid(characteristic["type"])

        if self.connection.database_is_partial:
            # The walk was cut short. Usable now, but persisting it would survive
            # a restart and be indistinguishable from a complete read. In memory
            # it stays at -1 so the next description update retries the read; a
            # cut-short walk is transient, so the retry is expected to complete.
            logger.debug("%s: not caching a truncated accessory database", self.name)
            self._accessories_state = AccessoriesState(Accessories.from_list(accessories), -1)
        elif self.connection.database_from_walk:
            # A walk infers the end of the database from a run of missing
            # instance ids, so it cannot prove it saw everything. In memory it
            # lives under the real config number -- at -1 every description
            # update would look like a config change, and a state-number bump
            # (which sleepy accessories send for every event) would trigger a
            # full re-walk instead of the catch-up poll. But it is *persisted*
            # under -1, a config number no advertisement carries, so a restart
            # restores entities from it and the first description update
            # re-reads for real.
            config_num = self.description.config_num if self.description else max(self.config_num, 0)
            self._accessories_state = AccessoriesState(Accessories.from_list(accessories), config_num)
            logger.debug("%s: caching the signature-walk database as always-stale", self.name)
            self.controller._char_cache.async_create_or_update_map(
                self.id, -1, self.accessories.serialize(), None, None
            )
        else:
            # max(..., 0): with no prior state config_num reports -1, which is
            # reserved above as the walk's always-stale marker; an authoritative
            # read must not be cached under it.
            self._accessories_state = AccessoriesState(
                Accessories.from_list(accessories), max(self.config_num, 0)
            )
            self._update_accessories_state_cache()

        return accessories

    async def get_primary_name(self) -> str:
        """Return the primary name of the device without enumerating it.

        Overrides the default, which reads the whole accessory database. This runs
        immediately after pairing, and on an accessory that does not implement the
        0x09 bulk read, enumerating means one request per instance id -- a walk
        that outlasts a controller's pairing dialog. Failing here loses the
        pairing while the accessory keeps it, leaving an orphan only a factory
        reset can clear.

        Zeroconf already told us the name, so use it and leave the database to be
        read later, off the dialog. Mirrors BlePairing.get_primary_name, which
        exists for the same reason.
        """
        # An empty Accessories() is truthy, so testing `not self.accessories`
        # alone would send a second call into the default implementation, where
        # the placeholder's empty list raises instead of enumerating.
        if self.description and (self.accessories is None or not list(self.accessories)):
            self._accessories_state = AccessoriesState(Accessories(), -1)
            return self.description.name
        return await super().get_primary_name()

    async def _process_config_changed(self, config_num: int) -> None:
        """Process a config change.

        This method is called when the config num changes.
        """
        # The instance ids may have moved, so the cached database cannot be reused.
        self.connection.invalidate_database()
        await self.list_accessories_and_characteristics()
        if self.connection.database_is_partial or self.connection.database_from_walk:
            # list_accessories_and_characteristics just declined to persist this
            # database as authoritative; stamping the accessory's real config
            # number here would overrule that and persist it as one. Listeners
            # still hear the change, and the stale marker makes the next
            # comparison re-read for real.
            for callback in self.config_changed_listeners:
                callback(self.config_num)
            return
        self._accessories_state = AccessoriesState(self._accessories_state.accessories, config_num)
        self._callback_and_save_config_changed(config_num)

    def _process_disconnected_events(self):
        """Process any events that happened while we were disconnected.

        We don't disconnect in COAP so there is no need to do anything here.
        """

    async def async_populate_accessories_state(
        self, force_update: bool = False, attempts: int | None = None
    ) -> bool:
        """Populate the state of all accessories.

        This method should try not to fetch all the accessories unless
        we know the config num is out of date or force_update is True
        """
        # `attempts` is how the controller says whether it is in a hurry: it is
        # 1 while Home Assistant is still starting up, and None once it is
        # running -- which is when a device has just been paired and losing the
        # session would orphan the pairing on the accessory. That is the one
        # place worth spending the full pair-verify budget.
        pair_verify_attempts = PAIR_VERIFY_ATTEMPTS if attempts is None else max(1, attempts)
        # `not self.accessories` alone cannot be trusted: get_primary_name's
        # placeholder is an empty-but-truthy Accessories(). Its config number of
        # -1 never matches an advertisement, so the comparison (the same one
        # BlePairing uses) is what guarantees the deferred first read happens.
        config_stale = self.description is not None and self.config_num != self.description.config_num
        if (
            not self.accessories
            or not list(self.accessories)
            or force_update
            or config_stale
            or self.connection.database_is_partial
        ):
            await self.list_accessories_and_characteristics(pair_verify_attempts)

    async def get_characteristics(
        self,
        characteristics: Iterable[tuple[int, int]],
    ) -> dict[tuple[int, int], dict[str, Any]]:
        await self._ensure_connected()
        return await self.connection.read_characteristics(characteristics)

    async def put_characteristics(
        self, characteristics: Iterable[tuple[int, int, Any]]
    ) -> dict[tuple[int, int], dict[str, Any]]:
        await self._ensure_connected()
        response_status = await self.connection.write_characteristics(characteristics)

        listener_update: dict[tuple[int, int], dict[str, Any]] = {}
        for characteristic in characteristics:
            aid, iid, value = characteristic
            accessory_chars = self.accessories.aid(aid).characteristics
            char = accessory_chars.iid(iid)
            if (
                response_status.get((aid, iid), HapStatusCode.SUCCESS) == HapStatusCode.SUCCESS
                and CharacteristicPermissions.paired_read in char.perms
            ):
                listener_update[(aid, iid)] = {"value": value}

        if listener_update:
            self._callback_listeners(listener_update)

        return response_status

    async def thread_provision(
        self,
        dataset: str,
    ) -> None:
        """Provision a device with Thread network credentials."""

    async def subscribe(self, characteristics):
        await self._ensure_connected()
        new_subs = await super().subscribe(set(characteristics))
        if len(new_subs) == 0:
            logger.debug("Nothing new to subscribe to, ignoring")
            return None
        return await self.connection.subscribe_to(list(new_subs))

    async def unsubscribe(self, characteristics):
        await self._ensure_connected()
        await super().unsubscribe(set(characteristics))
        return await self.connection.unsubscribe_from(characteristics)

    async def list_pairings(self):
        await self._ensure_connected(enumerate_database=False)
        pairing_tuples = await self.connection.list_pairings()
        pairings = list(
            map(
                lambda x: dict(
                    (
                        ("pairingId", x[0].decode()),
                        ("publicKey", x[1].hex()),
                        ("permissions", x[2]),
                        ("controllerType", (x[2] & 0x01 and "admin") or "regular"),
                    )
                ),
                pairing_tuples,
            )
        )
        return pairings

    async def remove_pairing(self, pairingId: str) -> bool:
        await self._ensure_connected(enumerate_database=False)
        if await self.connection.remove_pairing(pairingId):
            await self._shutdown_if_primary_pairing_removed(pairingId)
            return True
        return False
