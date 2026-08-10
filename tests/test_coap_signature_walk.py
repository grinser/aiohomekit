"""Grouping of signature-walk results into accessories/services.

The signature walk (used when 0x09 is unsupported) reads each characteristic's
signature, which carries the parent service but not an accessory id.
_database_from_signatures groups characteristics into services and starts a new
accessory at each Accessory Information service (0x3E), so single-accessory
devices yield one accessory and bridges yield one per accessory.
"""

import struct
import uuid

from aiohomekit.controller.ble.structs import Characteristic as CharacteristicTLV
from aiohomekit.controller.coap.connection import CoAPHomeKitConnection

from .coap_eve_harness import fake_owner

ACCESSORY_INFORMATION = 0x3E


def _sig(char_type, svc_type, svc_iid, fmt=0x04):
    return CharacteristicTLV(
        type=char_type,
        properties=0x10,
        presentation_format=struct.pack("<BxHxxx", fmt, 0x2700),
        service_type=svc_type.to_bytes(16, "little"),
        service_instance_id=svc_iid.to_bytes(2, "little"),
    ).encode()


def _conn():
    owner = fake_owner()
    return CoAPHomeKitConnection(owner, "::1", 5683)


def _char_iids(accessory):
    return sorted(c.instance_id for s in accessory.services for c in s.characteristics)


def test_single_accessory():
    sigs = {
        2: _sig(0x14, ACCESSORY_INFORMATION, 1),
        3: _sig(0x20, ACCESSORY_INFORMATION, 1),
        49: _sig(0x11, 0x96, 54),
        50: _sig(0x23, 0x96, 54),
    }
    db = _conn()._database_from_signatures(sigs)
    assert len(db.accessories) == 1
    acc = db.accessories[0]
    assert acc.instance_id == 1
    assert _char_iids(acc) == [2, 3, 49, 50]
    assert {s.instance_id: len(s.characteristics) for s in acc.services} == {1: 2, 54: 2}


def test_multi_accessory_split_on_second_accessory_information():
    sigs = {
        2: _sig(0x14, ACCESSORY_INFORMATION, 1),  # accessory 1
        3: _sig(0x20, ACCESSORY_INFORMATION, 1),
        4: _sig(0x11, 0x96, 5),
        10: _sig(0x14, ACCESSORY_INFORMATION, 8),  # accessory 2 starts here
        11: _sig(0x25, 0x43, 12),
    }
    db = _conn()._database_from_signatures(sigs)
    assert [a.instance_id for a in db.accessories] == [1, 2]
    a1, a2 = db.accessories
    assert _char_iids(a1) == [2, 3, 4]
    assert _char_iids(a2) == [10, 11]
    # the write path looks up by (aid, iid): iid 11 belongs to accessory 2 only
    assert db.find_characteristic_by_aid_iid(2, 11) is not None
    assert db.find_characteristic_by_aid_iid(1, 11) is None


def test_types_in_the_homekit_base_range_are_shortened():
    """Signatures carry full 128-bit UUIDs, but the 0x09 database reports the short
    form, and lookups like find_service_characteristic_by_type() are written against
    it. Without shortening, a walk-built database silently breaks those lookups --
    remove_pairing() and list_pairings() can never find the pairing service."""

    def full(short):
        """The full HomeKit UUID an accessory actually puts in a signature."""
        return uuid.UUID(f"{short:08X}-0000-1000-8000-0026BB765291").int

    pairing_service = 0x55
    pairing_pairings = 0x50
    vendor = uuid.UUID("E863F117-079E-48FF-8F27-9C2605A29F52").int
    sigs = {
        2: _sig(full(0x14), full(ACCESSORY_INFORMATION), 1),
        18: _sig(full(pairing_pairings), full(pairing_service), 17),
        32: _sig(vendor, full(0x96), 30),
    }
    db = _conn()._database_from_signatures(sigs)
    accessory = db.accessories[0]

    assert accessory.find_service_by_type(pairing_service) is not None
    found = accessory.find_service_characteristic_by_type(pairing_service, pairing_pairings)
    assert found is not None and found.instance_id == 18

    # Vendor UUIDs are outside the HomeKit base range and must survive intact.
    vendor_chars = [c for s in accessory.services for c in s.characteristics if c.type == vendor]
    assert len(vendor_chars) == 1


def test_accessories_that_reuse_service_instance_ids_are_still_split():
    # A bridge may number service instance ids per accessory, so accessory 2's
    # Accessory Information service can reuse an iid already seen in accessory 1.
    # The split must key on the service iid *changing*, not on it being unseen,
    # otherwise accessory 2 is merged into accessory 1's service of the same iid.
    sigs = {
        2: _sig(0x14, ACCESSORY_INFORMATION, 1),  # accessory 1, info service iid 1
        3: _sig(0x11, 0x96, 2),
        4: _sig(0x14, ACCESSORY_INFORMATION, 1),  # accessory 2 reuses info iid 1
        5: _sig(0x25, 0x43, 2),  # ... and reuses service iid 2
    }
    db = _conn()._database_from_signatures(sigs)
    assert [a.instance_id for a in db.accessories] == [1, 2]
    a1, a2 = db.accessories
    assert _char_iids(a1) == [2, 3]
    assert _char_iids(a2) == [4, 5]


def test_repeated_characteristics_in_one_accessory_information_service_do_not_split():
    # Several characteristics under the same Accessory Information service must
    # stay in one accessory (the split only triggers on a *new* 0x3E service).
    sigs = {
        2: _sig(0x14, ACCESSORY_INFORMATION, 1),
        3: _sig(0x20, ACCESSORY_INFORMATION, 1),
        4: _sig(0x21, ACCESSORY_INFORMATION, 1),
        5: _sig(0x11, 0x96, 54),
    }
    db = _conn()._database_from_signatures(sigs)
    assert len(db.accessories) == 1
    assert _char_iids(db.accessories[0]) == [2, 3, 4, 5]


async def test_access_controlled_characteristics_do_not_truncate_the_walk():
    """An access-controlled characteristic EXISTS -- we just may not read its
    signature. Counting it towards the gap run terminates the walk on a stretch
    of protected characteristics and reports the truncated result as complete,
    which is the defect this whole fallback is being fixed for.

    A first attempt at trimming the walk's cost did exactly that. The cost was
    real, but losing characteristics to buy it is the wrong trade.
    """
    from aiohomekit.controller.coap.pdu import PDUStatus

    from .coap_eve_harness import FakeEve, build_connection, signature

    ACCESSORY_INFORMATION, MISC = 0x3E, 0x96
    layout = {
        2: signature(0x14, ACCESSORY_INFORMATION, 1),
        3: signature(0x20, ACCESSORY_INFORMATION, 1),
        5: signature(0x23, ACCESSORY_INFORMATION, 1),
        # Beyond a run of 35 protected iids, so a walk that counts them as gaps
        # stops before ever seeing these.
        41: signature(0x10, MISC, 40),
        42: signature(0x11, MISC, 40),
    }

    class MostlyProtected(FakeEve):
        async def post(self, opcode, iid, data, timeout=16.0, expected_statuses=()):
            self.requests.append((opcode, iid))
            if iid in self.layout:
                return (len(self.layout[iid]), self.layout[iid])
            if 6 <= iid <= 40:
                return (0, PDUStatus.INSUFFICIENT_AUTHENTICATION)
            return (0, PDUStatus.INVALID_INSTANCE_ID)

    conn = build_connection(MostlyProtected(layout=layout))

    signatures, complete = await conn._signature_walk()

    assert sorted(signatures) == sorted(layout), (
        f"the walk lost {sorted(set(layout) - set(signatures))} to a run of access-controlled characteristics"
    )
    assert complete
