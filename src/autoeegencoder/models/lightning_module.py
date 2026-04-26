from __future__ import annotations

import numpy as np

from autoeegencoder.protocols.transforms import MethodArrays
from autoeegencoder.utils.metrics import balanced_accuracy
from autoeegencoder.utils.random import seed_everything

from .eegnet import EEGNetLiteFactory


def run_lightning_method(arrays: MethodArrays, cfg: dict, method: dict, run_seed: int) -> dict[str, float]:
    try:
        import torch
        from lightning import LightningModule, Trainer
        from lightning.pytorch.callbacks import EarlyStopping
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "Lightning backend requires torch and lightning. Install requirements.txt "
            "or run trainer=numpy_smoke with dataset=synthetic."
        ) from exc

    matmul_precision = cfg["trainer"].get("matmul_precision")
    if matmul_precision:
        torch.set_float32_matmul_precision(str(matmul_precision))
    seed_everything(int(run_seed))

    train_X = torch.tensor(arrays.train_X[:, None, :, :], dtype=torch.float32)
    train_y = torch.tensor(arrays.train_y, dtype=torch.long)
    train_r = torch.tensor(arrays.train_r, dtype=torch.float32)
    val_X = torch.tensor(arrays.val_X[:, None, :, :], dtype=torch.float32)
    val_y = torch.tensor(arrays.val_y, dtype=torch.long)
    val_r = torch.tensor(arrays.val_r, dtype=torch.float32)
    test_X = torch.tensor(arrays.test_X[:, None, :, :], dtype=torch.float32)
    test_y = torch.tensor(arrays.test_y, dtype=torch.long)
    test_r = torch.tensor(arrays.test_r, dtype=torch.float32)

    n_channels = arrays.train_X.shape[1]
    n_times = arrays.train_X.shape[2]
    n_classes = int(len(np.unique(arrays.train_y)))
    model_cfg = dict(cfg)
    model_cfg["model"] = dict(cfg["model"])
    model_cfg["model"]["adapter"] = dict(cfg["model"].get("adapter", {}))
    adapter_name = method.get("adapter", "none")
    model_cfg["model"]["adapter"]["enabled"] = adapter_name not in {None, "none"}
    model_cfg["model"]["adapter"]["variant"] = adapter_name

    class LitClassifier(LightningModule):
        def __init__(self):
            super().__init__()
            self.net = EEGNetLiteFactory.build(n_channels, n_times, n_classes, model_cfg)
            self.loss = torch.nn.CrossEntropyLoss()

        def forward(self, x, reliability=None):
            return self.net(x, reliability)

        def training_step(self, batch, batch_idx):
            x, y, r = batch
            loss = self.loss(self(x, r), y)
            self.log("train_loss", loss, prog_bar=False)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y, r = batch
            loss = self.loss(self(x, r), y)
            self.log("val_loss", loss, prog_bar=False)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=float(cfg["trainer"]["learning_rate"]),
                weight_decay=float(cfg["trainer"].get("weight_decay", 0.0)),
            )
            if cfg["trainer"].get("lr_scheduler") == "reduce_on_plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=float(cfg["trainer"].get("lr_factor", 0.5)),
                    patience=int(cfg["trainer"].get("lr_patience", 3)),
                    min_lr=float(cfg["trainer"].get("min_lr", 1.0e-5)),
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": cfg["trainer"].get("monitor", "val_loss"),
                    },
                }
            return optimizer

    batch_size = int(cfg["trainer"]["batch_size"])
    pin_memory = bool(cfg["trainer"].get("pin_memory", False))
    train_generator = torch.Generator().manual_seed(int(run_seed))
    train_loader = DataLoader(
        TensorDataset(train_X, train_y, train_r),
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=int(cfg["trainer"].get("num_workers", 0)),
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        TensorDataset(val_X, val_y, val_r),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg["trainer"].get("num_workers", 0)),
        pin_memory=pin_memory,
    )
    lit = LitClassifier()
    callbacks = []
    patience = int(cfg["trainer"].get("early_stopping_patience", 0))
    if patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=str(cfg["trainer"].get("monitor", "val_loss")),
                patience=patience,
                min_delta=float(cfg["trainer"].get("early_stopping_min_delta", 0.0)),
                mode="min",
            )
        )
    trainer = Trainer(
        accelerator=cfg["trainer"].get("accelerator", "cpu"),
        devices=cfg["trainer"].get("devices", 1),
        max_epochs=int(cfg["trainer"]["max_epochs"]),
        deterministic=bool(cfg["trainer"].get("deterministic", True)),
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=bool(cfg["trainer"].get("enable_progress_bar", False)),
        log_every_n_steps=int(cfg["trainer"].get("log_every_n_steps", 5)),
        precision=cfg["trainer"].get("precision", 32),
    )
    trainer.fit(lit, train_loader, val_loader)
    lit.eval()
    with torch.no_grad():
        device = lit.device
        logits = lit(test_X.to(device), test_r.to(device))
        pred = torch.argmax(logits, dim=1).cpu().numpy()
    y_true = test_y.cpu().numpy()
    return {
        "accuracy": float(np.mean(pred == y_true)),
        "balanced_accuracy": balanced_accuracy(y_true, pred),
    }
