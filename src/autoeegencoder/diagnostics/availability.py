from __future__ import annotations

from enum import Enum


class AvailabilityClass(str, Enum):
    D0A = "D0a"
    D0B = "D0b"
    D0C = "D0c"
    D1 = "D1"
    D2 = "D2"
    D3 = "D3"


_METHOD_ALLOWED = {
    "P0": {AvailabilityClass.D0A, AvailabilityClass.D0B},
    "P1": {AvailabilityClass.D0A, AvailabilityClass.D0B, AvailabilityClass.D1},
    "P2": {AvailabilityClass.D0A, AvailabilityClass.D0B, AvailabilityClass.D1, AvailabilityClass.D2},
}


def assert_method_availability(protocol_tier: str, availability: str, component: str) -> None:
    cls = AvailabilityClass(availability)
    allowed = _METHOD_ALLOWED.get(protocol_tier)
    if allowed is None:
        raise ValueError(f"Unknown protocol tier: {protocol_tier}")
    if cls not in allowed:
        raise ValueError(
            f"Diagnostic component '{component}' with availability {cls.value} "
            f"cannot enter method-side computation for {protocol_tier}."
        )

