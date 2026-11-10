"""Hourly poller: CarData telematicData -> Observation -> re-derived legs (ONTWERP.md §5).

Invoked every hour on the hour by EventBridge Scheduler. Outside the poll window from the web UI
(Instellingen, default 06-23 Europe/Amsterdam) it stops before any CarData call; the event
{"force": true} polls anyway (manual test); {"probe": true} asks BMW which energy descriptors
this car fills (3 calls, also outside the window). Order of business:
1. Load the token state from Secrets Manager; refresh when the access token expires within
   5 minutes. Refresh tokens rotate, so the new state is written back immediately.
2. Count every API call against a per-UTC-day quota in DynamoDB; refuse beyond QUOTA_LIMIT.
3. Fetch telematicData, store the Observation (idempotent per polled_at); with
   ENERGY_TRACKING also the latest energy state (own DynamoDB item, ENERGY#<vehicle>).
4. Re-derive + classify the CarData legs of yesterday and today.

Errors (refresh failure, API errors) are raised so the Lambda Errors metric and its alarm fire.
Hitting the quota is logged, not raised: retrying would not help until 00:00 UTC. So is "no car
linked yet" (link it on the Auto page of the web UI) and "choose a vehicle first".
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

from .. import cardata
from ..cardata import QuotaExceeded
from ..derive import AMSTERDAM, match_location
from ..model import LocationType
from ..process import rederive
from ..store import Store
from . import env, log_event


def quota_guard(store: Store, limit: int, now=lambda: datetime.now(UTC)):
    def before_api_call(method: str, path: str) -> None:
        used = store.increment_quota(now().date())
        if used > limit:
            raise QuotaExceeded(f"CarData quota: {used} > {limit} calls on {now().date()} UTC")

    return before_api_call


def poll(store: Store, client: cardata.CarDataClient, vehicle: str, now: datetime) -> dict:
    obs = client.observe(vehicle, polled_at=now)
    new = store.put_observations([obs])
    legs = rederive(store, vehicle, now)
    loc = match_location(obs.lat, obs.lon, store.locations())
    energy = {}
    if obs.energy:
        at_home = None if obs.lat is None else bool(loc and loc.type is LocationType.HOME)
        state = dataclasses.replace(
            obs.energy,
            location_id=loc.id if loc else None,
            at_home=at_home,
            lat=obs.lat,
            lon=obs.lon,
            loc_ts=obs.loc_ts,
        )
        store.put_energy_state(state)
        energy = {
            "soc_pct": state.soc_pct,
            "electric_range_km": state.electric_range_km,
            "charging_status": state.charging_status,
            "plugged_in": state.plugged_in,
        }
    return {
        **energy,
        "event": "poll",
        "vehicle": vehicle,
        "odometer_km": obs.odometer_km,
        "odometer_ts": obs.odometer_ts.isoformat(),
        "location": loc.id if loc else ("unknown" if obs.lat is not None else None),
        "new_observation": bool(new),
        "legs_recent": len(legs),
        "business_km_recent": sum(int(leg.business_km) for leg in legs),
        "quota_used": store.quota_used(now.date()),
    }


def handler(
    event, context, *, store: Store | None = None, token_store=None, now: datetime | None = None
) -> dict:
    from ..store.dynamodb import DynamoDbStore

    store = store or DynamoDbStore(env("TABLE_NAME", "kmlog-data"))
    token_store = token_store or cardata.SecretsManagerTokenStore(env("SECRET_ID", "kmlog/cardata"))
    now = now or datetime.now(UTC)
    hour = now.astimezone(AMSTERDAM).hour
    event = event or {}
    if not (event.get("force") or event.get("probe")) and not store.settings().polls_at(hour):
        result = {"event": "outside_window", "hour": hour}
        log_event(**result)
        return result
    client = cardata.CarDataClient(
        token_store,
        before_api_call=quota_guard(store, int(env("QUOTA_LIMIT", "40"))),
        descriptors=cardata.descriptors_for(env("ENERGY_TRACKING", "0") == "1"),
    )
    if not client.linked:
        result = {"event": "not_linked"}
    elif not client.state.get("vin") and client.state.get("vehicles"):
        result = {"event": "vehicle_not_chosen"}
    else:
        result = None
    try:
        if event.get("probe") and not result:
            result = {"event": "probe", **client.probe(cardata.PROBE_DESCRIPTORS)}
        result = result or poll(store, client, env("VEHICLE_ID", "car"), now)
        if client.energy_rejected:
            result["energy_rejected"] = True
    except QuotaExceeded as e:
        result = {"event": "quota_exceeded", "message": str(e)}
    log_event(**result)
    return result
