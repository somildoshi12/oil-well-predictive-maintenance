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


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),  # raw logits; sigmoid at predict time
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def _train_loop(model, loader, val_loader, optimizer, criterion,
                epochs, patience, mode, checkpoint_dir, ckpt_name,
                start_epoch=0, best_val_init=None, best_state_init=None,
                desc="Model"):
    best_val = best_val_init if best_val_init is not None else (
        float("-inf") if mode == "max" else float("inf")
    )
    best_state = best_state_init
    patience_count = 0
    t0 = time.time()

    pbar = tqdm(range(start_epoch, epochs), desc=desc,
                unit="epoch", dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for epoch in pbar:
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_losses.append(criterion(model(xb), yb).item())
        val_metric = float(np.mean(val_losses))

        improved = (val_metric > best_val) if mode == "max" else (val_metric < best_val - 1e-6)
        if improved:
            best_val = val_metric
            patience_count = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pbar.set_postfix(val=f"{val_metric:.4f}", patience=patience_count,
                             best="*", elapsed=f"{time.time()-t0:.0f}s")
        else:
            patience_count += 1
            pbar.set_postfix(val=f"{val_metric:.4f}", patience=patience_count,
                             elapsed=f"{time.time()-t0:.0f}s")

        if checkpoint_dir and (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_state": best_state,
                "optimizer_state": optimizer.state_dict(),
                "best_val": best_val,
            }, os.path.join(checkpoint_dir, ckpt_name))

        if patience_count >= patience:
            pbar.set_postfix(val=f"{val_metric:.4f}", status="EARLY STOP")
            pbar.close()
            if checkpoint_dir:
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "best_state": best_state,
                    "optimizer_state": optimizer.state_dict(),
                    "best_val": best_val,
                }, os.path.join(checkpoint_dir, ckpt_name))
            break

    elapsed = time.time() - t0
    print(f"  {desc} done in {elapsed:.1f}s  |  best_val={best_val:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    return model


def _load_loop_checkpoint(model, optimizer, checkpoint_dir, ckpt_name):
    if not checkpoint_dir:
        return 0, None, None
    path = os.path.join(checkpoint_dir, ckpt_name)
    if not os.path.exists(path):
        return 0, None, None
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    resume_epoch = ckpt["epoch"] + 1
    print(f"  [Checkpoint] Resuming {ckpt_name} from epoch {resume_epoch}")
    return resume_epoch, ckpt["best_val"], ckpt["best_state"]


class FailurePredictor:
    def __init__(self, input_dim: int, checkpoint_dir: str = None):
        self.input_dim = input_dim
        self.checkpoint_dir = checkpoint_dir
        self.classifier = MLPClassifier(input_dim).to(DEVICE)
        self.regressor = MLPRegressor(input_dim).to(DEVICE)

    def fit_classifier(self, X_train, y_train, X_val, y_val,
                       class_weight=None, epochs=50, batch_size=256):
        pos_weight = None
        if class_weight and 0 in class_weight and 1 in class_weight:
            ratio = float(class_weight[1]) / float(class_weight[0])
            pos_weight = torch.tensor([ratio], dtype=torch.float32, device=DEVICE)

        criterion = (nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                     if pos_weight is not None else nn.BCEWithLogitsLoss())

        Xt = torch.FloatTensor(X_train).to(DEVICE)
        yt = torch.FloatTensor(y_train).to(DEVICE)
        Xv = torch.FloatTensor(X_val).to(DEVICE)
        yv = torch.FloatTensor(y_val).to(DEVICE)

        loader     = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(Xv, yv), batch_size=batch_size * 4)
        optimizer  = torch.optim.Adam(self.classifier.parameters(), lr=1e-3)

        start_epoch, best_val, best_state = _load_loop_checkpoint(
            self.classifier, optimizer, self.checkpoint_dir, "classifier_ckpt.pt"
        )

        _train_loop(
            self.classifier, loader, val_loader, optimizer, criterion,
            epochs=epochs, patience=5, mode="min",
            checkpoint_dir=self.checkpoint_dir, ckpt_name="classifier_ckpt.pt",
            start_epoch=start_epoch, best_val_init=best_val, best_state_init=best_state,
            desc="Classifier",
        )

    def fit_regressor(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=256):
        criterion  = nn.HuberLoss()
        Xt = torch.FloatTensor(X_train).to(DEVICE)
        yt = torch.FloatTensor(y_train.astype(float)).to(DEVICE)
        Xv = torch.FloatTensor(X_val).to(DEVICE)
        yv = torch.FloatTensor(y_val.astype(float)).to(DEVICE)

        loader     = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(Xv, yv), batch_size=batch_size * 4)
        optimizer  = torch.optim.Adam(self.regressor.parameters(), lr=1e-3)

        start_epoch, best_val, best_state = _load_loop_checkpoint(
            self.regressor, optimizer, self.checkpoint_dir, "regressor_ckpt.pt"
        )

        _train_loop(
            self.regressor, loader, val_loader, optimizer, criterion,
            epochs=epochs, patience=5, mode="min",
            checkpoint_dir=self.checkpoint_dir, ckpt_name="regressor_ckpt.pt",
            start_epoch=start_epoch, best_val_init=best_val, best_state_init=best_state,
            desc="Regressor",
        )

    def predict_failure(self, X: np.ndarray) -> np.ndarray:
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(torch.FloatTensor(X).to(DEVICE))
            return torch.sigmoid(logits).cpu().numpy()

    def predict_days(self, X: np.ndarray) -> np.ndarray:
        self.regressor.eval()
        with torch.no_grad():
            return self.regressor(torch.FloatTensor(X).to(DEVICE)).cpu().numpy()

    def save(self, clf_path: str, reg_path: str):
        torch.save({"state_dict": self.classifier.state_dict(), "input_dim": self.input_dim}, clf_path)
        torch.save({"state_dict": self.regressor.state_dict(), "input_dim": self.input_dim}, reg_path)
        print(f"  Classifier saved → {clf_path}")
        print(f"  Regressor  saved → {reg_path}")

    @classmethod
    def load(cls, clf_path: str, reg_path: str):
        clf_data = torch.load(clf_path, map_location=DEVICE, weights_only=True)
        reg_data = torch.load(reg_path, map_location=DEVICE, weights_only=True)
        input_dim = clf_data["input_dim"]
        predictor = cls(input_dim)
        predictor.classifier.load_state_dict(clf_data["state_dict"])
        predictor.classifier.eval()
        predictor.regressor.load_state_dict(reg_data["state_dict"])
        predictor.regressor.eval()
        return predictor
