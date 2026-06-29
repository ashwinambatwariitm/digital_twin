"""
Trainer
───────
Full training loop for Mini-GraphCast.
Supports:
  - Mixed precision (fp16) for RTX 3050
  - Early stopping
  - Cosine LR scheduler
  - Checkpoint saving / loading
  - Live loss logging
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import logging
from datetime import datetime

log = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        graph,
        config: dict,
        resume_from: str = None,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.ckpt_dir     = Path(config["paths"]["checkpoints"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        cfg = config["training"]

        # ── Device ───────────────────────────────────────────
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        log.info(f"Training on: {self.device}")
        if self.device.type == "cuda":
            log.info(f"GPU: {torch.cuda.get_device_name(0)}")
            log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        self.model = model.to(self.device)

        # ── Graph on device ──────────────────────────────────
        self.edge_index = graph.edge_index.to(self.device)
        self.edge_attr  = graph.edge_attr.to(self.device)

        # ── Optimizer ────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = float(cfg["learning_rate"]),
            weight_decay = float(cfg["weight_decay"]),
        )

        # ── Scheduler ────────────────────────────────────────
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max  = int(cfg["epochs"]),
            eta_min = float(cfg["learning_rate"]) * 0.01,
        )

        # ── Mixed precision (fp16) for RTX 3050 ──────────────
        self.use_amp = cfg["mixed_precision"] and self.device.type == "cuda"
        self.scaler  = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        log.info(f"Mixed precision (fp16): {self.use_amp}")

        # ── Loss ─────────────────────────────────────────────
        self.criterion = WeightedMSELoss()

        # ── State ────────────────────────────────────────────
        self.epochs         = int(cfg["epochs"])
        self.grad_clip      = float(cfg["grad_clip"])
        self.early_patience = int(cfg["early_stopping"])
        self.history        = {"train_loss": [], "val_loss": [], "lr": []}
        self.best_val_loss  = float("inf")
        self.patience_count = 0
        self.start_epoch    = 0

        if resume_from:
            self._load_checkpoint(resume_from)

    def train(self) -> dict:
        """Run full training loop. Returns history dict."""
        log.info(f"Starting training for {self.epochs} epochs...")

        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self._train_epoch(epoch)
            val_loss   = self._val_epoch(epoch)
            lr         = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(lr)

            self.scheduler.step()

            # ── Checkpoint ───────────────────────────────────
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_count = 0
                self._save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.patience_count += 1
                if epoch % 5 == 0:
                    self._save_checkpoint(epoch, val_loss, is_best=False)

            print(
                f"Epoch {epoch+1:03d}/{self.epochs} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"{'✅ Best' if is_best else f'⏳ Patience {self.patience_count}/{self.early_patience}'}"
            )

            # ── Early stopping ────────────────────────────────
            if self.patience_count >= self.early_patience:
                log.info(f"Early stopping at epoch {epoch+1}")
                break

        # Save training history
        history_path = self.ckpt_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        log.info(f"History saved to {history_path}")

        return self.history

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch+1}", leave=False)
        for x, y in pbar:
            x = x.to(self.device)   # (B, S, N, C)
            y = y.to(self.device)   # (B, P, N, 3)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred = self.model(x, self.edge_index, self.edge_attr)
                loss = self.criterion(pred, y)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred = self.model(x, self.edge_index, self.edge_attr)
                loss = self.criterion(pred, y)

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        ckpt = {
            "epoch":         epoch + 1,
            "model_state":   self.model.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "scheduler":     self.scheduler.state_dict(),
            "scaler":        self.scaler.state_dict(),
            "val_loss":      val_loss,
            "best_val_loss": self.best_val_loss,
            "config":        self.config,
            "timestamp":     datetime.now().isoformat(),
        }

        path = self.ckpt_dir / ("best_model.pt" if is_best else f"epoch_{epoch+1:03d}.pt")
        torch.save(ckpt, str(path))

    def _load_checkpoint(self, path: str):
        log.info(f"Resuming from checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch    = ckpt["epoch"]
        self.best_val_loss  = ckpt.get("best_val_loss", float("inf"))
        log.info(f"Resumed from epoch {self.start_epoch}")


def load_best_model(model, config: dict) -> nn.Module:
    """Load best checkpoint into model."""
    ckpt_path = Path(config["paths"]["checkpoints"]) / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best model found at {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    log.info(f"Loaded best model (val_loss={ckpt['val_loss']:.4f})")
    return model


# ── Loss ─────────────────────────────────────────────────────────────────────

class WeightedMSELoss(nn.Module):
    """
    MSE loss with per-channel weights.
    Rainfall is harder to predict — give it more weight.
    """

    def __init__(self, weights=(2.0, 1.0, 1.0)):
        super().__init__()
        self.register_buffer(
            "weights",
            torch.tensor(weights, dtype=torch.float32)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, P, N, 3)
        diff = (pred - target) ** 2                    # (B, P, N, 3)
        diff = diff * self.weights.to(diff.device).view(1, 1, 1, 3)    # weight per channel
        return diff.mean()