import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AnomalyDetector:
    def __init__(self, input_dim: int, threshold_percentile: float = 95.0,
                 checkpoint_dir: str = None):
        self.model = Autoencoder(input_dim).to(DEVICE)
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        self.input_dim = input_dim
        self.checkpoint_dir = checkpoint_dir

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _ckpt_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "autoencoder_ckpt.pt")

    def _save_checkpoint(self, epoch: int, optimizer, best_loss: float, best_state: dict):
        if self.checkpoint_dir is None:
            return
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "best_state": best_state,
            "optimizer_state": optimizer.state_dict(),
            "best_loss": best_loss,
            "input_dim": self.input_dim,
            "threshold_percentile": self.threshold_percentile,
        }, self._ckpt_path())

    def _load_checkpoint(self, optimizer):
        path = self._ckpt_path()
        if self.checkpoint_dir and os.path.exists(path):
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            self.model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            resume_epoch = ckpt["epoch"] + 1
            print(f"  [Checkpoint] Resuming autoencoder from epoch {resume_epoch}")
            return resume_epoch, ckpt["best_loss"], ckpt["best_state"]
        return 0, float("inf"), None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X_normal: np.ndarray, epochs: int = 50, batch_size: int = 256):
        X_t = torch.FloatTensor(X_normal).to(DEVICE)
        loader = DataLoader(TensorDataset(X_t, X_t), batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        start_epoch, best_loss, best_state = self._load_checkpoint(optimizer)
        patience, patience_count = 5, 0
        t0 = time.time()

        pbar = tqdm(range(start_epoch, epochs), desc="Autoencoder",
                    unit="epoch", dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        for epoch in pbar:
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(X_normal)

            improved = epoch_loss < best_loss - 1e-6
            if improved:
                best_loss = epoch_loss
                patience_count = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                pbar.set_postfix(loss=f"{epoch_loss:.5f}", patience=patience_count,
                                 best="*", elapsed=f"{time.time()-t0:.0f}s")
            else:
                patience_count += 1
                pbar.set_postfix(loss=f"{epoch_loss:.5f}", patience=patience_count,
                                 elapsed=f"{time.time()-t0:.0f}s")

            # Checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch, optimizer, best_loss, best_state)

            if patience_count >= patience:
                pbar.set_postfix(loss=f"{epoch_loss:.5f}", status="EARLY STOP")
                pbar.close()
                self._save_checkpoint(epoch, optimizer, best_loss, best_state)
                break

        if best_state:
            self.model.load_state_dict(best_state)

        errors = self._reconstruction_errors(X_normal)
        self.threshold = float(np.percentile(errors, self.threshold_percentile))
        elapsed = time.time() - t0
        print(f"  Autoencoder done in {elapsed:.1f}s  |  "
              f"threshold (p{self.threshold_percentile}): {self.threshold:.6f}")
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(DEVICE)
            recon = self.model(X_t).cpu().numpy()
        return np.mean((X - recon) ** 2, axis=1)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        return self._reconstruction_errors(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.anomaly_score(X) > self.threshold).astype(int)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, model_path: str, threshold_path: str):
        torch.save(self.model.state_dict(), model_path)
        np.save(threshold_path, np.array([self.threshold, self.input_dim]))
        print(f"  Autoencoder saved → {model_path}")

    @classmethod
    def load(cls, model_path: str, threshold_path: str):
        arr = np.load(threshold_path)
        threshold = float(arr[0])
        input_dim = int(arr[1])
        detector = cls(input_dim)
        detector.model.load_state_dict(
            torch.load(model_path, map_location=DEVICE, weights_only=True)
        )
        detector.model.eval()
        detector.threshold = threshold
        return detector
