from __future__ import annotations

ENHANCED_ADAPTERS = {
    "qca_enhanced",
    "qca_d0b_local_only",
    "qca_d0a_source_ref_only",
    "qca_random_reliability",
    "qca_shuffled_reliability",
    "qca_adapter_only",
    "qca_shuffle_adapter_only",
    "qca_shuffle_norm_only",
    "generic_film",
}


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("torch is required for the Lightning training backend.") from exc
    return torch, nn


class EEGNetLiteFactory:
    @staticmethod
    def build(n_channels: int, n_times: int, n_classes: int, cfg: dict):
        torch, nn = require_torch()

        class EEGNetLite(nn.Module):
            def __init__(self):
                super().__init__()
                f1 = int(cfg["model"].get("n_temporal_filters", 8))
                d = int(cfg["model"].get("depth_multiplier", 2))
                f2 = int(cfg["model"].get("separable_filters", 16))
                kernel = int(cfg["model"].get("kernel_length", 32))
                dropout = float(cfg["model"].get("dropout", 0.25))
                self.adapter_enabled = bool(cfg["model"].get("adapter", {}).get("enabled", False))
                adapter_cfg = cfg["model"].get("adapter", {})
                self.adapter_variant = str(adapter_cfg.get("variant", "none"))
                self.adapter_alpha = float(adapter_cfg.get("alpha", 0.1))
                self.feature_alpha = float(adapter_cfg.get("feature_alpha", 0.08))
                adapter_hidden = int(adapter_cfg.get("hidden_dim", max(16, n_channels)))
                global_dim = int(adapter_cfg.get("global_feature_dim", 5))
                self.reliability_dim = n_channels + global_dim
                self.enhanced_adapter = self.adapter_variant in ENHANCED_ADAPTERS

                self.temporal = nn.Sequential(
                    nn.Conv2d(1, f1, kernel_size=(1, kernel), padding=(0, kernel // 2), bias=False),
                    nn.BatchNorm2d(f1),
                )
                self.depthwise = nn.Sequential(
                    nn.Conv2d(f1, f1 * d, kernel_size=(n_channels, 1), groups=f1, bias=False),
                    nn.BatchNorm2d(f1 * d),
                    nn.ELU(),
                    nn.AvgPool2d(kernel_size=(1, 4)),
                    nn.Dropout(dropout),
                )
                self.separable = nn.Sequential(
                    nn.Conv2d(f1 * d, f1 * d, kernel_size=(1, 16), padding=(0, 8), groups=f1 * d, bias=False),
                    nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False),
                    nn.BatchNorm2d(f2),
                    nn.ELU(),
                    nn.AvgPool2d(kernel_size=(1, 8)),
                    nn.Dropout(dropout),
                )
                with torch.no_grad():
                    dummy = torch.zeros(1, 1, n_channels, n_times)
                    n_features = self._features(dummy).shape[1]
                self.classifier = nn.Linear(n_features, n_classes)
                if self.adapter_enabled:
                    gate_in_dim = self.reliability_dim if self.enhanced_adapter else n_channels
                    self.channel_gate = nn.Sequential(
                        nn.Linear(gate_in_dim, adapter_hidden),
                        nn.ReLU(),
                        nn.Linear(adapter_hidden, n_channels),
                    )
                    nn.init.zeros_(self.channel_gate[-1].weight)
                    nn.init.zeros_(self.channel_gate[-1].bias)
                    if self.enhanced_adapter:
                        self.feature_gate = nn.Sequential(
                            nn.Linear(self.reliability_dim, adapter_hidden),
                            nn.ReLU(),
                            nn.Linear(adapter_hidden, f2),
                        )
                        nn.init.zeros_(self.feature_gate[-1].weight)
                        nn.init.zeros_(self.feature_gate[-1].bias)

            def _features(self, x):
                h = self.temporal(x)
                h = self.depthwise(h)
                h = self.separable(h)
                return h.flatten(start_dim=1)

            def forward(self, x, reliability=None):
                if self.adapter_enabled and reliability is not None:
                    if self.enhanced_adapter:
                        summary = reliability
                        channel_reliability = reliability[:, :n_channels]
                    else:
                        summary = reliability[:, :n_channels]
                        channel_reliability = reliability[:, :n_channels]
                    gate = 1.0 + self.adapter_alpha * torch.tanh(self.channel_gate(summary))
                    gate = gate * (0.75 + 0.25 * channel_reliability)
                    x = x * gate[:, None, :, None]
                h = self.temporal(x)
                h = self.depthwise(h)
                h = self.separable(h)
                if self.adapter_enabled and reliability is not None and self.enhanced_adapter:
                    feature_gate = 1.0 + self.feature_alpha * torch.tanh(self.feature_gate(reliability))
                    h = h * feature_gate[:, :, None, None]
                h = h.flatten(start_dim=1)
                return self.classifier(h)

        return EEGNetLite()
