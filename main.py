# ──────────────────────────────────────────────────────────────
# 0. Imports & constants
# ──────────────────────────────────────────────────────────────
import pandas as pd, numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXCEL  = r"C:\Users\aryan\OneDrive\Documents\sohams_dop\Data_Prep\CNN_Input_Labeled_0_1_only.xlsx"

SEG_LEN, N_FEATS = 170, 5          # 170 time-steps × 5 distance bins
BATCH, EPOCHS, LR = 64, 60, 1e-3   # training hyper-params

# ──────────────────────────────────────────────────────────────
# 1. Load Excel → (samples, 170, 5)  +  labels
# ──────────────────────────────────────────────────────────────
df = pd.read_excel(EXCEL)

X_full = df.iloc[:, :N_FEATS].values.reshape(-1, SEG_LEN, N_FEATS).astype(np.float32)
y_full = df.iloc[:, -1].values.astype(np.float32).reshape(-1, SEG_LEN)[:, 0]

# global z-score normalisation (feature-wise)
mean = X_full.reshape(-1, N_FEATS).mean(0, keepdims=True)
std  = X_full.reshape(-1, N_FEATS).std (0, keepdims=True) + 1e-8
X_full = (X_full - mean) / std

# ──────────────────────────────────────────────────────────────
# 2. 70 / 15 / 15 stratified split
# ──────────────────────────────────────────────────────────────
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, tmp_idx = next(sss1.split(X_full, y_full))

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_idx, test_idx = next(sss2.split(X_full[tmp_idx], y_full[tmp_idx]))

def to_dl(x, y, shuffle=False):
    ds = TensorDataset(torch.tensor(x), torch.tensor(y))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

train_dl = to_dl(X_full[train_idx], y_full[train_idx], shuffle=True)
val_dl   = to_dl(X_full[tmp_idx][val_idx],  y_full[tmp_idx][val_idx])
test_dl  = to_dl(X_full[tmp_idx][test_idx], y_full[tmp_idx][test_idx])

# ──────────────────────────────────────────────────────────────
# 3. Minimal 1-D ResNet
# ──────────────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, in_c, out_c, k=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, k, padding=k//2),
            nn.BatchNorm1d(out_c), nn.ReLU(),
            nn.Conv1d(out_c, out_c, k, padding=k//2),
            nn.BatchNorm1d(out_c))
        self.act  = nn.ReLU()
        self.skip = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))

class ResNet1D(nn.Module):
    def __init__(self, n_feats=5):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv1d(n_feats, 64, 7, padding=3),
                                    nn.BatchNorm1d(64), nn.ReLU())
        self.b1 = Block(64, 128)
        self.b2 = Block(128, 256)
        self.b3 = Block(256, 256)          # extra capacity
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(256, 1)      # single logit (binary)
    def forward(self, x):
        x = x.transpose(1, 2)              # (B, C=5, T=170)
        x = self.stem(x)
        x = self.b1(x); x = self.b2(x); x = self.b3(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)

model = ResNet1D().to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
crit  = nn.BCEWithLogitsLoss()

# scheduler without `verbose`
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='min', factor=0.5, patience=5)

lr_now = lambda: opt.param_groups[0]['lr']

# ──────────────────────────────────────────────────────────────
# 4. Train / validate
# ──────────────────────────────────────────────────────────────
def run_epoch(dl, train=False):
    all_p, all_t, total_loss = [], [], 0.
    model.train() if train else model.eval()
    for xb, yb in dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if train: opt.zero_grad()
        logits = model(xb)
        loss   = crit(logits, yb)
        if train:
            loss.backward()
            opt.step()
        total_loss += loss.item() * len(xb)
        all_p.append(torch.sigmoid(logits).detach().cpu())
        all_t.append(yb.cpu())
    p = torch.cat(all_p); t = torch.cat(all_t)
    acc = accuracy_score(t, (p > 0.5))
    return total_loss / len(dl.dataset), acc, p, t

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc, _, _ = run_epoch(train_dl, train=True)
    val_loss, val_acc, _, _ = run_epoch(val_dl)
    scheduler.step(val_loss)

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
          f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
          f"LR {lr_now():.2e}")

# ──────────────────────────────────────────────────────────────
# 5. Final test evaluation
# ──────────────────────────────────────────────────────────────
_, _, p, t = run_epoch(test_dl)
test_acc = accuracy_score(t, (p > 0.5))
test_roc = roc_auc_score(t,  p)

print(f"\n🎯  Test accuracy {test_acc:.3f} | ROC-AUC {test_roc:.3f}")

cm = confusion_matrix(t, (p > 0.5))
plt.imshow(cm, cmap="Blues"); plt.title("Confusion matrix"); plt.colorbar()
plt.xticks([0,1]); plt.yticks([0,1]); plt.xlabel("Pred"); plt.ylabel("True")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j],
                 ha='center', va='center',
                 color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.show()
