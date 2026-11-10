import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from kmlog.model import (
    DayNote,
    Leg,
    LegSource,
    LegStatus,
    Location,
    LocationType,
    Observation,
    Rate,
    Route,
    Vehicle,
    VehicleKind,
)
from kmlog.store.codec import dumps, loads
from kmlog.store.sqlite import SqliteStore


@pytest.fixture(params=["sqlite", "dynamodb"])
def store(request):
    if request.param == "sqlite":
        s = SqliteStore()
    else:
        from kmlog.store.dynamodb import DynamoDbStore

        request.getfixturevalue("dynamodb_table")
        s = DynamoDbStore("kmlog-data")
    yield s
    s.close()


T0 = datetime(2026, 10, 7, 6, 0, tzinfo=UTC)


def test_stamdata_roundtrip(store):
    vehicle = Vehicle("car", "BMW i4", "AB-123-C", VehicleKind.CAR, True, date(2026, 1, 1))
    store.put_vehicle(vehicle)
    store.put_rate(Rate(2026, 25))
    store.put_rate(Rate(2026, 25))  # replace, not duplicate
    store.put_location(Location("home", "Thuis", LocationType.HOME, 52.0, 4.5))
    store.put_route(Route("a4", "home", "client", "A4", 70, True))
    assert store.vehicle("car") == vehicle
    assert store.vehicle("nope") is None
    assert store.rate(2026) == Rate(2026, 25)
    assert store.rate(2025) is None
    assert store.locations()[0].type is LocationType.HOME
    assert store.routes()[0].is_default


def test_observations_are_idempotent_and_ordered(store):
    obs = [
        Observation("car", T0 + timedelta(hours=h), 100 + h, T0, 52.0, 4.5, T0, {"raw": h})
        for h in (2, 0, 1)
    ]
    assert store.put_observations(obs) == 3
    assert store.put_observations(obs) == 0
    got = store.observations("car")
    assert [o.odometer_km for o in got] == [100, 101, 102]
    assert got[0] == obs[1]
    assert got[0].polled_at.tzinfo is not None
    window = store.observations("car", start=T0 + timedelta(hours=1), end=T0 + timedelta(hours=2))
    assert [o.odometer_km for o in window] == [101]
    assert store.observation_before("car", T0 + timedelta(hours=2)).odometer_km == 101
    assert store.observation_before("car", T0 + timedelta(days=3)).odometer_km == 102
    assert store.observation_before("car", T0) is None
    assert store.observation_before("other", T0 + timedelta(days=3)) is None


def test_observations_stored_with_motion_fields_still_load():
    # written 2026-10-12/13 while the container asked isMoving/ignition (always null)
    obs = Observation("car", T0, 100, T0, 52.0, 4.5, T0)
    data = json.loads(dumps(obs)) | {"moving": None, "ignition_on": None}
    assert loads(Observation, json.dumps(data)) == obs


def test_naive_datetimes_are_rejected(store):
    naive = datetime(2026, 10, 7, 6, 0)
    with pytest.raises(ValueError):
        store.put_observations([Observation("car", naive, 1, naive, None, None, None)])


def _leg(day: int, source=LegSource.CARDATA, **kw) -> Leg:
    return Leg(
        id=f"car#{source.value}#{day}",
        vehicle="car",
        date=date(2026, 10, day),
        distance_km=Decimal(70),
        source=source,
        **kw,
    )


def test_leg_roundtrip_keeps_all_fields(store):
    leg = _leg(
        7,
        departed_after=T0,
        arrived_at=T0 + timedelta(hours=1),
        from_latlon=(52.0, 4.5),
        flags=["weekend_or_holiday"],
        business_km=Decimal(70),
        status=LegStatus.NEEDS_CONFIRMATION,
    )
    store.put_leg(leg)
    assert store.legs("car", date(2026, 10, 1), date(2026, 10, 31)) == [leg]


def test_replace_legs_only_touches_its_source_and_range(store):
    store.replace_legs(
        "car", LegSource.MANUAL, date(2026, 10, 1), date(2026, 10, 31),
        [_leg(5, LegSource.MANUAL), _leg(7, LegSource.MANUAL)],
    )  # fmt: skip
    store.put_leg(_leg(8))
    store.replace_legs(
        "car", LegSource.MANUAL, date(2026, 10, 1), date(2026, 10, 31),
        [_leg(5, LegSource.MANUAL)],
    )  # fmt: skip
    got = store.legs("car", date(2026, 10, 1), date(2026, 10, 31))
    assert [(leg.date.day, leg.source) for leg in got] == [
        (5, LegSource.MANUAL),
        (8, LegSource.CARDATA),
    ]
    with pytest.raises(ValueError):
        store.replace_legs(
            "car", LegSource.CARDATA, date(2026, 10, 1), date(2026, 10, 6), [_leg(8)]
        )


def test_day_notes(store):
    notes = [
        DayNote("car", date(2026, 4, 27), "Koningsdag"),
        DayNote("car", date(2026, 5, 25), "x"),
    ]
    store.replace_day_notes("car", date(2026, 1, 1), date(2026, 12, 31), notes)
    store.replace_day_notes("car", date(2026, 1, 1), date(2026, 12, 31), notes)
    assert store.day_notes("car", date(2026, 4, 1), date(2026, 4, 30)) == notes[:1]


def test_quota_counter_per_day(store):
    day = date(2026, 10, 11)
    assert store.quota_used(day) == 0
    assert [store.increment_quota(day) for _ in range(3)] == [1, 2, 3]
    assert store.quota_used(day) == 3
    assert store.quota_used(day + timedelta(days=1)) == 0


def test_replace_places(store):
    store.put_location(Location("old", "Oud", LocationType.PRIVATE, 1.0, 2.0))
    store.put_route(Route("r", "old", "home", "x", 5))
    store.replace_places([Location("home", "Thuis", LocationType.HOME, 52.0, 4.5)], [])
    assert [loc.id for loc in store.locations()] == ["home"]
    assert store.routes() == []


def test_answers_and_overrides(store):
    t = datetime(2026, 10, 12, 18, 0, tzinfo=UTC)
    store.put_answer("car", "car#20261012T0730", "detour", t)
    store.put_answer("car", "car#20261012T0730", "private_stop", t)  # replace
    store.put_answer("car", "car#20261013T0730", "business", t)
    store.put_answer("other", "x", "business", t)
    assert store.answers("car") == {
        "car#20261012T0730": "private_stop",
        "car#20261013T0730": "business",
    }
    store.delete_answer("car", "car#20261013T0730")
    assert list(store.answers("car")) == ["car#20261012T0730"]

    store.put_override("car", "car#20261012T0730", Decimal(70), "file A12")
    assert store.overrides("car") == {"car#20261012T0730": (Decimal(70), "file A12")}
    store.delete_override("car", "car#20261012T0730")
    assert store.overrides("car") == {}


def test_notified(store):
    t = datetime(2026, 10, 12, 18, 0, tzinfo=UTC)
    store.mark_notified("car", ["seg:a", "missed:2026-10-12"], t)
    store.mark_notified("car", ["seg:a"], t)
    assert store.notified("car") == {"seg:a", "missed:2026-10-12"}
    assert store.notified("other") == set()


def test_delete_leg(store):
    store.put_leg(_leg(8, LegSource.MANUAL))
    store.put_leg(_leg(9, LegSource.MANUAL))
    store.delete_leg(_leg(8, LegSource.MANUAL))
    assert [leg.date.day for leg in store.legs("car", date(2026, 10, 1), date(2026, 10, 31))] == [9]


def test_login_tokens_are_single_use_and_expire(store):
    now = datetime(2026, 10, 12, 18, 0, tzinfo=UTC)
    store.put_login_token("h1", now + timedelta(minutes=15))
    store.put_login_token("h2", now - timedelta(seconds=1))
    assert store.consume_login_token("h1", now)
    assert not store.consume_login_token("h1", now)  # used
    assert not store.consume_login_token("h2", now)  # expired
    assert not store.consume_login_token("nope", now)


def test_counter(store):
    exp = datetime(2026, 10, 12, 19, 0, tzinfo=UTC)
    assert [store.increment_counter("login:2026101218", exp) for _ in range(3)] == [1, 2, 3]
    assert store.increment_counter("login:2026101219", exp) == 1


def test_legs_imported_from_a_spreadsheet_load_as_manual():
    # databases from before the spreadsheet importer was removed
    old = dumps(_leg(5)).replace('"source": "cardata"', '"source": "excel_2026", "legacy": true')
    old = old.replace(json.loads(old)["id"], "car#excel#2026-10-05")
    leg = loads(Leg, old)
    assert leg.source is LegSource.MANUAL and leg.date == date(2026, 10, 5)
    assert leg.imported and not _leg(5, LegSource.MANUAL).imported


def test_energy_state_roundtrip_and_older_poll_does_not_overwrite(store):
    from kmlog.model import EnergyState

    assert store.energy_state("car") is None
    new = EnergyState(
        "car", T0 + timedelta(hours=2), soc_pct=40, charging_status="X", plugged_in=False
    )
    store.put_energy_state(new)
    assert store.energy_state("car") == new
    store.put_energy_state(EnergyState("car", T0, soc_pct=90))
    assert store.energy_state("car") == new
    newer = EnergyState("car", T0 + timedelta(hours=3), soc_pct=80, at_home=True, location_id="h")
    store.put_energy_state(newer)
    assert store.energy_state("car") == newer
    assert store.energy_state("other") is None


def test_energy_item_has_flat_attributes(aws, dynamodb_table):
    from kmlog.model import EnergyState
    from kmlog.store.dynamodb import DynamoDbStore

    s = DynamoDbStore(dynamodb_table)
    s.put_energy_state(
        EnergyState(
            "car",
            T0,
            soc_pct=40,
            electric_range_km=18,
            charging_status="NOCHARGING",
            plug_status="DISCONNECTED",
            plugged_in=False,
            at_home=True,
            location_id="home",
            reported_at=T0,
        )
    )
    item = s.table.get_item(Key={"PK": "ENERGY#car", "SK": "LATEST"})["Item"]
    assert item["soc_pct"] == 40 and item["electric_range_km"] == 18
    assert item["plugged_in"] is False and item["at_home"] is True
    assert item["charging_status"] == "NOCHARGING" and item["location_id"] == "home"
    assert item["polled_at"] == T0.isoformat() and item["reported_at"] == T0.isoformat()
    assert "fuel_level_pct" not in item and "total_range_km" not in item


def test_energy_item_position_is_decimal_and_omitted_when_absent(aws, dynamodb_table):
    from decimal import Decimal

    from kmlog.model import EnergyState
    from kmlog.store.dynamodb import DynamoDbStore

    s = DynamoDbStore(dynamodb_table)
    with_pos = EnergyState("car", T0, soc_pct=40, lat=52.08940277777778, lon=5.1, loc_ts=T0)
    s.put_energy_state(with_pos)
    item = s.table.get_item(Key={"PK": "ENERGY#car", "SK": "LATEST"})["Item"]
    assert item["lat"] == Decimal("52.08940277777778") and item["lon"] == Decimal("5.1")
    assert str(item["lat"]) == "52.08940277777778"
    assert item["loc_ts"] == T0.isoformat()
    assert s.energy_state("car") == with_pos
    s.put_energy_state(EnergyState("car", T0 + timedelta(hours=1), soc_pct=41))
    item = s.table.get_item(Key={"PK": "ENERGY#car", "SK": "LATEST"})["Item"]
    assert not {"lat", "lon", "loc_ts"} & item.keys()


def test_energy_state_without_position_fields_still_loads():
    from kmlog.model import EnergyState

    old = json.loads(dumps(EnergyState("car", T0, soc_pct=50)))
    for k in ("lat", "lon", "loc_ts"):
        old.pop(k)
    e = loads(EnergyState, json.dumps(old))
    assert (e.lat, e.lon, e.loc_ts) == (None, None, None)


def test_observation_with_energy_roundtrips_and_one_without_still_loads(store):
    from kmlog.model import EnergyState

    energy = EnergyState("car", T0, soc_pct=50, plugged_in=True, reported_at=T0)
    obs = Observation("car", T0, 100, T0, 52.0, 4.5, T0, None, energy)
    store.put_observations([obs])
    assert store.observations("car") == [obs]
    old = json.loads(dumps(Observation("car", T0, 100, T0, 52.0, 4.5, T0)))
    old.pop("energy")  # written by a version without the field
    assert loads(Observation, json.dumps(old)).energy is None
    newer = json.loads(dumps(obs)) | {"future_field": 1}
    assert loads(Observation, json.dumps(newer)).energy == energy
