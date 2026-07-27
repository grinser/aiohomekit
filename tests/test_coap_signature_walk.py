"""Grouping of signature-walk results into accessories/services.

The signature walk (used when 0x09 is unsupported) reads each characteristic's
signature, which carries the parent service but not an accessory id.
_database_from_signatures groups characteristics into services and starts a new
accessory at each Accessory Information service (0x3E), so single-accessory
devices yield one accessory and bridges yield one per accessory.
"""

import struct

from aiohomekit.controller.ble.structs import Characteristic as CharacteristicTLV
from aiohomekit.controller.coap.connection import CoAPHomeKitConnection

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
    owner = type("Owner", (), {"accessories": None, "event_received": lambda *a: None})()
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
