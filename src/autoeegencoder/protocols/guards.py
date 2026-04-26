from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProtocolGuard:
    protocol_tier: str
    events: list[dict[str, str]] = field(default_factory=list)

    @property
    def allow_target_aggregate(self) -> bool:
        return self.protocol_tier in {"P1", "P2"}

    def record(self, action: str, availability: str, detail: str) -> None:
        self.events.append(
            {
                "protocol_tier": self.protocol_tier,
                "action": action,
                "availability": availability,
                "detail": detail,
            }
        )

    def require_target_aggregate_allowed(self, detail: str) -> None:
        if not self.allow_target_aggregate:
            raise ValueError(f"P0 protocol violation: target aggregate requested for {detail}")
        self.record("use_target_aggregate", "D1", detail)

    def rows(self, dataset_name: str, method_name: str, subject: int) -> list[dict[str, str]]:
        return [
            {
                "dataset": dataset_name,
                "method": method_name,
                "test_subject": str(subject),
                **event,
            }
            for event in self.events
        ]

