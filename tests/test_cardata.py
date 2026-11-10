import base64
import hashlib
import json
from datetime import UTC, datetime

import boto3
import pytest

from kmlog import cardata
from kmlog.cardata import (
    ApiError,
    AuthError,
    CarDataClient,
    FileTokenStore,
    SecretsManagerTokenStore,
    parse_observation,
)

# shape of a real response (2026-10-10)
TELEMATIC = {
    "telematicData": {
        cardata.LATITUDE_DESCRIPTOR: {
            "timestamp": "2026-10-10T14:01:06.000Z",
            "unit": "degrees",
            "value": "52.08940277777778",
        },
        cardata.ODOMETER_DESCRIPTOR: {
            "timestamp": "2026-10-10T14:01:06.000Z",
            "unit": "km",
            "value": "23456",
        },
        cardata.LONGITUDE_DESCRIPTOR: {
            "timestamp": "2026-10-10T14:01:06.000Z",
            "unit": "degrees",
            "value": "5.110025000000001",
        },
    }
}
NOW = 1_760_000_000.0


class Resp:
    def __init__(self, status: int, body):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakeSession:
    """Scripted responses per (method, url-suffix); records every call."""

    def __init__(self, routes: dict[tuple[str, str], list[Resp]]):
        self.routes = {k: list(v) for k, v in routes.items()}
        self.calls: list[tuple[str, str, dict]] = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        for (m, suffix), responses in self.routes.items():
            if m == method and url.endswith(suffix):
                return responses.pop(0) if len(responses) > 1 else responses[0]
        raise AssertionError(f"unexpected {method} {url}")

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)


class MemoryStore:
    def __init__(self, state=None):
        self.state = dict(state or {})
        self.saves = 0

    def load(self):
        return dict(self.state)

    def save(self, state):
        self.state = dict(state)
        self.saves += 1


def logged_in(**extra):
    return MemoryStore(
        {
            "client_id": "cid",
            "access_token": "at",
            "refresh_token": "rt",
            "expires_at": NOW + 3600,
            **extra,
        }
    )


def client(store, routes, calls=None):
    return CarDataClient(
        store,
        session=FakeSession(routes),
        clock=lambda: NOW,
        before_api_call=(lambda m, p: calls.append((m, p))) if calls is not None else None,
    )


def test_parse_observation_matches_real_response():
    polled = datetime(2026, 10, 10, 16, 13, tzinfo=UTC)
    obs = parse_observation(TELEMATIC["telematicData"], "car", polled)
    assert obs.odometer_km == 23456
    assert obs.odometer_ts == datetime(2026, 10, 10, 14, 1, 6, tzinfo=UTC)
    assert (obs.lat, obs.lon) == (52.08940277777778, 5.110025000000001)
    assert obs.loc_ts == obs.odometer_ts
    assert obs.polled_at == polled


def test_parse_without_position_and_with_decimal_odometer():
    tele = {cardata.ODOMETER_DESCRIPTOR: {"value": "23456.0", "timestamp": "2026-10-10T14:01:06Z"}}
    obs = parse_observation(tele, "car", datetime(2026, 10, 10, tzinfo=UTC))
    assert (obs.odometer_km, obs.lat, obs.lon, obs.loc_ts) == (23456, None, None, None)


def test_parse_without_odometer_fails():
    with pytest.raises(cardata.CarDataError):
        parse_observation({}, "car", datetime(2026, 10, 10, tzinfo=UTC))


def test_first_observe_resolves_vin_and_container_once():
    store, calls = logged_in(), []
    c = client(
        store,
        {
            ("GET", "/customers/vehicles/mappings"): [
                Resp(200, [{"vin": "WBY1", "mappingType": "PRIMARY"}])
            ],
            ("POST", "/customers/containers"): [Resp(201, {"containerId": "E00O050G311QH"})],
            ("GET", "/customers/vehicles/WBY1/telematicData"): [Resp(200, TELEMATIC)],
        },
        calls,
    )
    obs = c.observe("car")
    assert obs.odometer_km == 23456
    assert (store.state["vin"], store.state["container_id"]) == ("WBY1", "E00O050G311QH")
    c.observe("car")
    assert [p for _, p in calls] == [
        "/customers/vehicles/mappings",
        "/customers/containers",
        "/customers/vehicles/WBY1/telematicData",
        "/customers/vehicles/WBY1/telematicData",
    ]
    _, _, kwargs = c.http.calls[-1]
    assert kwargs["params"] == {"containerId": "E00O050G311QH"}
    assert kwargs["headers"]["x-version"] == "v1"


def test_mappings_wrapped_in_object_also_work():
    c = client(
        logged_in(),
        {
            ("GET", "/mappings"): [
                Resp(200, {"mappings": [{"vin": "A", "mappingType": "SECONDARY"}]})
            ]
        },
    )
    with pytest.raises(cardata.CarDataError, match="PRIMARY"):
        c.vin()


def test_expiring_token_is_refreshed_and_rotation_persisted_before_the_call():
    store = logged_in(
        expires_at=NOW + 60, vin="V", container_id="C", container_descriptors=cardata.DESCRIPTORS
    )
    c = client(
        store,
        {
            ("POST", cardata.TOKEN_URL): [
                Resp(200, {"access_token": "at2", "refresh_token": "rt2", "expires_in": 3600})
            ],
            ("GET", "/customers/vehicles/V/telematicData"): [Resp(200, TELEMATIC)],
        },
    )
    c.observe("car")
    assert store.state["access_token"] == "at2"
    assert store.state["refresh_token"] == "rt2"
    assert store.state["expires_at"] == NOW + 3600
    method, url, kwargs = c.http.calls[0]
    assert url == cardata.TOKEN_URL
    assert kwargs["data"]["grant_type"] == "refresh_token"
    assert kwargs["data"]["refresh_token"] == "rt"
    assert c.http.calls[1][2]["headers"]["Authorization"] == "Bearer at2"


def test_failed_refresh_raises_auth_error():
    c = client(
        logged_in(expires_at=NOW - 1),
        {("POST", cardata.TOKEN_URL): [Resp(400, {"error": "invalid_grant"})]},
    )
    with pytest.raises(AuthError):
        c.observe("car")


def test_not_logged_in():
    with pytest.raises(AuthError):
        client(MemoryStore({"client_id": "cid"}), {}).observe("car")


def test_api_error_carries_status():
    c = client(
        logged_in(vin="V", container_id="C", container_descriptors=cardata.DESCRIPTORS),
        {("GET", "/telematicData"): [Resp(403, {"exveErrorId": "CU-429"})]},
    )
    with pytest.raises(ApiError) as exc:
        c.observe("car")
    assert exc.value.status == 403
    assert "CU-429" in exc.value.body


def test_quota_hook_can_stop_the_call():
    class Stop(Exception):
        pass

    def hook(method, path):
        raise Stop

    c = CarDataClient(
        logged_in(vin="V", container_id="C", container_descriptors=cardata.DESCRIPTORS),
        session=FakeSession({}),
        clock=lambda: NOW,
        before_api_call=hook,
    )
    with pytest.raises(Stop):
        c.observe("car")
    assert c.http.calls == []


def test_device_login_uses_pkce():
    store = MemoryStore({"client_id": "cid"})
    c = client(
        store,
        {
            ("POST", cardata.DEVICE_CODE_URL): [
                Resp(
                    200,
                    {
                        "device_code": "dc",
                        "user_code": "ABCD",
                        "verification_uri": "https://example/verify",
                        "interval": 5,
                        "expires_in": 300,
                    },
                )
            ],
            ("POST", cardata.TOKEN_URL): [
                Resp(400, {"error": "authorization_pending"}),
                Resp(200, {"access_token": "at", "refresh_token": "rt", "expires_in": 3600}),
            ],
        },
    )
    login = c.start_device_login()
    c.complete_device_login(login, sleep=lambda s: None)
    dc_call = c.http.calls[0][2]["data"]
    assert dc_call["response_type"] == "device_code"
    assert dc_call["code_challenge_method"] == "S256"
    assert dc_call["scope"] == "authenticate_user openid cardata:api:read"
    expected = base64.urlsafe_b64encode(hashlib.sha256(login.code_verifier.encode()).digest())
    assert dc_call["code_challenge"] == expected.rstrip(b"=").decode()
    assert c.http.calls[-1][2]["data"]["code_verifier"] == login.code_verifier
    assert store.state["access_token"] == "at"


def test_file_store_reads_the_probe_format(tmp_path):
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps({"access_token": "a", "obtained_at": 100.0, "expires_in": 3599}))
    state = FileTokenStore(path).load()
    assert state["expires_at"] == 3699.0
    FileTokenStore(path).save(state)
    assert path.stat().st_mode & 0o777 == 0o600


def test_secrets_manager_store_roundtrip(aws):
    sm = boto3.client("secretsmanager")
    sm.create_secret(Name="kmlog/cardata")
    store = SecretsManagerTokenStore("kmlog/cardata", sm)
    store.save({"client_id": "cid", "access_token": "a", "expires_at": 1.0, "id_token": "x"})
    assert store.load() == {"client_id": "cid", "access_token": "a", "expires_at": 1.0}


def test_container_with_old_descriptors_is_replaced_once():
    store, calls = logged_in(vin="V", container_id="OLD"), []
    c = client(
        store,
        {
            ("POST", "/customers/containers"): [Resp(201, {"containerId": "NEW"})],
            ("DELETE", "/customers/containers/OLD"): [Resp(204, "")],
            ("GET", "/customers/vehicles/V/telematicData"): [Resp(200, TELEMATIC)],
        },
        calls,
    )
    c.observe("car")
    c.observe("car")
    assert store.state["container_id"] == "NEW"
    assert store.state["container_descriptors"] == cardata.DESCRIPTORS
    assert calls == [
        ("POST", "/customers/containers"),
        ("DELETE", "/customers/containers/OLD"),
        ("GET", "/customers/vehicles/V/telematicData"),
        ("GET", "/customers/vehicles/V/telematicData"),
    ]
    assert c.http.calls[0][2]["json"]["technicalDescriptors"] == cardata.DESCRIPTORS
    assert c.http.calls[2][2]["params"] == {"containerId": "NEW"}


def test_failed_delete_of_old_container_does_not_stop_the_poll():
    store = logged_in(vin="V", container_id="OLD")
    c = client(
        store,
        {
            ("POST", "/customers/containers"): [Resp(201, {"containerId": "NEW"})],
            ("DELETE", "/customers/containers/OLD"): [Resp(404, {"error": "gone"})],
            ("GET", "/customers/vehicles/V/telematicData"): [Resp(200, TELEMATIC)],
        },
    )
    assert c.observe("car").odometer_km == 23456
    assert store.state["container_id"] == "NEW"


# --- energy state -------------------------------------------------------------------------

E = cardata.ENERGY_DESCRIPTORS
ENERGY_TELE = {
    **TELEMATIC["telematicData"],
    E["soc"]: {"value": "63.0", "unit": "%", "timestamp": "2026-10-10T14:00:00.000Z"},
    E["electric_range"]: {"value": "21.6", "unit": "km", "timestamp": "2026-10-10T14:02:00Z"},
    E["total_range"]: {"value": "512", "unit": "km", "timestamp": "2026-10-10T14:00:00Z"},
    E["charging_status"]: {"value": "CHARGINGACTIVE", "timestamp": "2026-10-10T14:00:00Z"},
    E["plug"]: {"value": "CONNECTED", "timestamp": "2026-10-10T14:00:00Z"},
    E["time_remaining"]: {"value": "95", "unit": "min", "timestamp": "2026-10-10T14:00:00Z"},
}
POLLED = datetime(2026, 10, 10, 16, 13, tzinfo=UTC)


def test_parse_energy_full_response():
    obs = parse_observation(ENERGY_TELE, "car", POLLED)
    e = obs.energy
    assert (e.soc_pct, e.electric_range_km, e.total_range_km, e.fuel_level_pct) == (
        63,
        22,
        512,
        None,
    )
    assert (e.charging_status, e.plug_status, e.plugged_in) == ("CHARGINGACTIVE", "CONNECTED", True)
    assert e.charge_time_remaining_min == 95
    assert e.reported_at == datetime(2026, 10, 10, 14, 2, tzinfo=UTC)
    assert e.polled_at == POLLED and (e.location_id, e.at_home) == (None, None)


def test_parse_energy_all_null_and_absent():
    nulls = {name: {"value": None, "timestamp": None} for name in E.values()}
    e = cardata.parse_energy(nulls, "car", POLLED)
    assert e is not None
    assert (e.soc_pct, e.charging_status, e.plugged_in, e.reported_at) == (None, None, None, None)
    assert cardata.parse_energy(TELEMATIC["telematicData"], "car", POLLED) is None
    assert parse_observation(TELEMATIC["telematicData"], "car", POLLED).energy is None


def test_parse_energy_soc_falls_back_to_displayed_and_keeps_unknown_strings_raw():
    tele = {
        E["soc_displayed"]: {"value": "55", "timestamp": "2026-10-10T14:00:00Z"},
        E["charging_status"]: {"value": "SOMETHING_NEW"},
        E["plug"]: {"value": "weird"},
    }
    e = cardata.parse_energy(tele, "car", POLLED)
    assert (e.soc_pct, e.charging_status, e.plug_status, e.plugged_in) == (
        55,
        "SOMETHING_NEW",
        "weird",
        None,
    )


@pytest.mark.parametrize(
    ("value", "plugged"),
    [
        ("CONNECTED", True),
        ("DISCONNECTED", False),
        ("NOT_CONNECTED", False),
        ("NOTCONNECTED", False),
        ("connected", True),
        ("UNKNOWN", None),
        (None, None),
    ],
)
def test_plugged_in_mapping(value, plugged):
    e = cardata.parse_energy({E["plug"]: {"value": value}}, "car", POLLED)
    assert e.plugged_in is plugged


def test_descriptors_for():
    assert cardata.descriptors_for(False) == cardata.DESCRIPTORS
    on = cardata.descriptors_for(True)
    assert on[: len(cardata.DESCRIPTORS)] == cardata.DESCRIPTORS
    assert set(E.values()) <= set(on)
    assert set(on) <= set(cardata.PROBE_DESCRIPTORS)


def test_descriptors_param_drives_container_and_toggling_replaces_it_once():
    on = cardata.descriptors_for(True)
    store = logged_in(vin="V", container_id="OLD", container_descriptors=cardata.DESCRIPTORS)
    routes = {
        ("POST", "/customers/containers"): [Resp(201, {"containerId": "NEW"})],
        ("DELETE", "/customers/containers/OLD"): [Resp(204, "")],
        ("GET", "/customers/vehicles/V/telematicData"): [Resp(200, TELEMATIC)],
    }
    calls = []
    c = CarDataClient(
        store,
        session=FakeSession(routes),
        clock=lambda: NOW,
        before_api_call=lambda m, p: calls.append(m),
        descriptors=on,
    )
    c.observe("car")
    c.observe("car")
    assert calls == ["POST", "DELETE", "GET", "GET"]
    assert c.http.calls[0][2]["json"]["technicalDescriptors"] == on
    assert store.state["container_descriptors"] == on
    # default client (energy off) with an energy container: replaced back
    c2 = client(
        store,
        {
            ("POST", "/customers/containers"): [Resp(201, {"containerId": "N2"})],
            ("DELETE", "/customers/containers/NEW"): [Resp(204, "")],
            ("GET", "/customers/vehicles/V/telematicData"): [Resp(200, TELEMATIC)],
        },
    )
    c2.observe("car")
    assert store.state["container_descriptors"] == cardata.DESCRIPTORS


def _probe_routes(get):
    return {
        ("POST", "/customers/containers"): [Resp(201, {"containerId": "P"})],
        ("DELETE", "/customers/containers/P"): [Resp(204, "")],
        ("GET", "/customers/vehicles/V/telematicData"): [get],
    }


def test_probe_uses_three_calls_and_leaves_the_real_container_alone():
    store, calls = logged_in(vin="V", container_id="REAL"), []
    c = client(store, _probe_routes(Resp(200, ENERGY_TELE)), calls)
    result = c.probe()
    assert calls == [
        ("POST", "/customers/containers"),
        ("GET", "/customers/vehicles/V/telematicData"),
        ("DELETE", "/customers/containers/P"),
    ]
    assert c.http.calls[0][2]["json"]["name"] == cardata.PROBE_CONTAINER_NAME
    assert c.http.calls[1][2]["params"] == {"containerId": "P"}
    assert store.state["container_id"] == "REAL"
    assert result["filled"][E["soc"]]["value"] == "63.0"
    assert E["plug"] in result["filled"] and E["soc_displayed"] in result["empty"]
    text = json.dumps(result)
    assert "52.089" not in text and "5.110" not in text  # no position
    assert '"V"' not in text and "at" not in result  # no VIN


def test_probe_deletes_the_container_when_the_read_fails():
    calls = []
    c = client(logged_in(vin="V"), _probe_routes(Resp(500, {"error": "boom"})), calls)
    with pytest.raises(ApiError):
        c.probe()
    assert calls[-1] == ("DELETE", "/customers/containers/P")


CU402 = Resp(400, {"exveErrorId": "CU-402", "exveErrorMsg": "Telematic key is invalid"})


def _fallback_client(store, extra_routes=None, descriptors=None, calls=None):
    routes = {
        ("POST", "/customers/containers"): [CU402, Resp(201, {"containerId": "BASE"})],
        ("DELETE", "/customers/containers/OLD"): [Resp(204, "")],
        ("GET", "/customers/vehicles/V/telematicData"): [Resp(200, TELEMATIC)],
        **(extra_routes or {}),
    }
    return CarDataClient(
        store,
        session=FakeSession(routes),
        clock=lambda: NOW,
        before_api_call=(lambda m, p: calls.append(m)) if calls is not None else None,
        descriptors=descriptors if descriptors is not None else cardata.descriptors_for(True),
    )


def test_cu402_falls_back_to_base_container_and_records_rejection():
    on = cardata.descriptors_for(True)
    store, calls = logged_in(vin="V", container_id="OLD", container_descriptors=[]), []
    c = _fallback_client(store, calls=calls)
    assert c.observe("car").odometer_km == 23456
    assert calls == ["POST", "POST", "DELETE", "GET"]
    posts = [k["json"]["technicalDescriptors"] for m, _, k in c.http.calls if m == "POST"]
    assert posts == [on, cardata.DESCRIPTORS]
    assert store.state["container_id"] == "BASE"
    assert store.state["container_descriptors"] == cardata.DESCRIPTORS
    assert store.state["rejected_descriptors"] == on
    assert c.energy_rejected


def test_after_rejection_the_next_poll_makes_no_extra_post():
    on = cardata.descriptors_for(True)
    store = logged_in(
        vin="V",
        container_id="BASE",
        container_descriptors=cardata.DESCRIPTORS,
        rejected_descriptors=on,
    )
    calls = []
    c = _fallback_client(store, calls=calls)
    c.observe("car")
    c.observe("car")
    assert calls == ["GET", "GET"]
    assert c.energy_rejected


def test_changed_descriptor_list_retries_once():
    old = cardata.descriptors_for(True)
    new = [*cardata.DESCRIPTORS, E["soc"]]
    store = logged_in(
        vin="V",
        container_id="OLD",
        container_descriptors=cardata.DESCRIPTORS,
        rejected_descriptors=old,
    )
    calls = []
    c = _fallback_client(
        store,
        {("POST", "/customers/containers"): [Resp(201, {"containerId": "FIXED"})]},
        descriptors=new,
        calls=calls,
    )
    assert not c.energy_rejected
    c.observe("car")
    assert store.state["container_id"] == "FIXED"
    assert store.state["container_descriptors"] == new
    assert calls == ["POST", "DELETE", "GET"]


def test_cu402_on_base_descriptors_still_raises():
    store = logged_in(vin="V")
    c = _fallback_client(store, descriptors=list(cardata.DESCRIPTORS))
    with pytest.raises(ApiError):
        c.observe("car")
    assert "rejected_descriptors" not in store.state


def test_other_400_does_not_fall_back():
    store = logged_in(vin="V")
    c = _fallback_client(
        store, {("POST", "/customers/containers"): [Resp(400, {"error": "other"})]}
    )
    with pytest.raises(ApiError):
        c.observe("car")


def test_probe_cu402_returns_error_instead_of_raising():
    calls = []
    routes = {("POST", "/customers/containers"): [CU402]}
    c = client(logged_in(vin="V"), routes, calls)
    result = c.probe()
    assert result["filled"] == {} and result["empty"] == []
    assert result["error"].startswith("CU-402")
    assert result["descriptors"] == cardata.PROBE_DESCRIPTORS
    assert calls == [("POST", "/customers/containers")]


def test_probe_other_post_error_raises():
    c = client(logged_in(vin="V"), {("POST", "/customers/containers"): [Resp(500, {})]})
    with pytest.raises(ApiError):
        c.probe()
