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

"""Callers that only need a database to exist must not each build one.

On a reconnect every coroutine waiting on the connection is released at once,
and each checks `info is None` before any of them takes the enumeration lock.
The lock serialises them but does not combine them, so without a re-check
inside it, N waiters cost N full enumerations -- on an accessory that answers
0x09 that is N database reads plus N sweeps of every readable characteristic.
"""

from __future__ import annotations

import asyncio

from aiohomekit.controller.coap.connection import CoAPHomeKitConnection

from .coap_eve_harness import EVE_LAYOUT, FakeEve, build_connection

WAITERS = 4


def _bulk_read_connection() -> CoAPHomeKitConnection:
    """An accessory that answers 0x09, i.e. one that re-reads on every
    enumeration and therefore shows the amplification most clearly."""
    encoded = CoAPHomeKitConnection._database_from_signatures(None, EVE_LAYOUT).encode()
    return build_connection(FakeEve(gatt="body", gatt_body=encoded))


async def test_concurrent_readiness_checks_cost_one_enumeration():
    conn = _bulk_read_connection()
    enumerations = 0
    release = asyncio.Event()
    original = conn._read_gatt_database

    async def _counted(*args, **kwargs):
        nonlocal enumerations
        enumerations += 1
        await release.wait()
        return await original(*args, **kwargs)

    conn._read_gatt_database = _counted

    waiters = [asyncio.create_task(conn.get_accessory_info(only_if_missing=True)) for _ in range(WAITERS)]
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(*waiters)

    assert enumerations == 1, (
        f"{WAITERS} callers that only needed a database to exist caused {enumerations} enumerations"
    )
    assert conn.info is not None


async def test_an_explicit_caller_still_gets_a_fresh_read():
    """only_if_missing is opt-in. list_accessories_and_characteristics and the
    config-changed handler are asking for current state, not for any state."""
    conn = _bulk_read_connection()
    await conn.get_accessory_info()

    enumerations = 0
    original = conn._read_gatt_database

    async def _counted(*args, **kwargs):
        nonlocal enumerations
        enumerations += 1
        return await original(*args, **kwargs)

    conn._read_gatt_database = _counted
    await conn.get_accessory_info()

    assert enumerations == 1
