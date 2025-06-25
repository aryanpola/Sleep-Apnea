# ──────────────────────────────────────────────────────────────
# 0. Imports & constants
# ──────────────────────────────────────────────────────────────
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
import matplotlib.pyplot as plt

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
EXCEL   = r"C:\Users\aryan\OneDrive\Documents\sohams_dop\Data_Prep\CNN_Input_Labeled_0_1_only.xlsx"
SEG, F  = 170, 5
BATCH   = 64
EPOCHS  = 60
LR      = 1e-3

# ──────────────────────────────────────────────────────────────
# 1. Load Excel  →  X (samples, 170, 5) , y (samples,)
# ──────────────────────────────────────────────────────────────
df  = pd.read_excel(EXCEL)
X   = df.iloc[:, :F].values.reshape(-1, SEG, F).astype(np.float32)
y   = df.iloc[:, -1].values.astype(np.float32).reshape(-1, SEG)[:, 0]

# global z-score per feature
μ = X.reshape(-1, F).mean(0, keepdims=True)
σ = X.reshape(-1, F).std (0, keepdims=True) + 1e-8
X = (X - μ) / σ

# ──────────────────────────────────────────────────────────────
# 2. Helper utilities
# ──────────────────────────────────────────────────────────────
def make_loader(x, y, shuffle=False):
    ds = TensorDataset(torch.tensor(x), torch.tensor(y))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

class Block(nn.Module):
    def __init__(self, in_c, out_c, k=7):
        super().__init__()
        self.net  = nn.Sequential(
            nn.Conv1d(in_c, out_c, k, padding=k//2),
            nn.BatchNorm1d(out_c), nn.ReLU(),
            nn.Conv1d(out_c, out_c, k, padding=k//2),
            nn.BatchNorm1d(out_c))
        self.act  = nn.ReLU()
        self.skip = nn.Conv1d(in_c, out_c, 1) if in_c!=out_c else nn.Identity()
    def forward(self,x): return self.act(self.net(x)+self.skip(x))

class ResNet1D(nn.Module):
    def __init__(self, n_feats=F):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(n_feats, 64, 7, padding=3),
                                  nn.BatchNorm1d(64), nn.ReLU())
        self.b1 = Block(64,128); self.b2 = Block(128,256); self.b3 = Block(256,256)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(256, 1)                       # single logit
    def forward(self,x):
        x = x.transpose(1,2)
        x = self.stem(x); x = self.b1(x); x = self.b2(x); x = self.b3(x)
        return self.head(self.pool(x).squeeze(-1)).squeeze(-1)

def metrics_from_probs(probs, targets):
    preds = (probs > 0.5).astype(int)
    return dict(
        acc  = accuracy_score (targets, preds),
        prec = precision_score(targets, preds, zero_division=0),
        rec  = recall_score   (targets, preds),
        sens = recall_score   (targets, preds),            # synonym
        f1   = f1_score       (targets, preds),
        auc  = roc_auc_score  (targets, probs)
    )

def train_one_split(X_tr, y_tr, X_val, y_val, verbose=False):
    model     = ResNet1D().to(DEVICE)
    opt       = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched     = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.5, 5)
    loss_fn   = nn.BCEWithLogitsLoss()

    tr_dl = make_loader(X_tr,  y_tr,  shuffle=True)
    val_dl = make_loader(X_val, y_val)

    def run(dl, train=False):
        model.train() if train else model.eval()
        all_p, all_t, tot = [], [], 0.0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if train: opt.zero_grad()
            out  = model(xb); loss = loss_fn(out, yb)
            if train: loss.backward(); opt.step()
            tot += loss.item()*len(xb)
            all_p.append(torch.sigmoid(out).detach().cpu())
            all_t.append(yb.cpu())
        return tot/len(dl.dataset), torch.cat(all_p).numpy(), torch.cat(all_t).numpy()

    for ep in range(1, EPOCHS+1):
        train_loss, _, _ = run(tr_dl, train=True)
        val_loss, val_p, val_t = run(val_dl)
        sched.step(val_loss)
        if verbose and ep % 10 == 0:
            print(f"  ep{ep:02d} train{train_loss:.3f} val{val_loss:.3f}")

    return metrics_from_probs(val_p, val_t), model

# ──────────────────────────────────────────────────────────────
# 3. Outer 15 % test split (never touched during CV)
# ──────────────────────────────────────────────────────────────
outer = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
dev_idx, test_idx = next(outer.split(X, y))
X_dev, y_dev = X[dev_idx], y[dev_idx]
X_test, y_test = X[test_idx], y[test_idx]

# ──────────────────────────────────────────────────────────────
# 4. 5-fold cross-validation on the 85 % dev set
# ──────────────────────────────────────────────────────────────
print("── 5-Fold Cross-Validation on 85 % dev set ──")
fold_metrics = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr, va) in enumerate(skf.split(X_dev, y_dev), 1):
    m, _ = train_one_split(X_dev[tr], y_dev[tr], X_dev[va], y_dev[va])
    fold_metrics.append(m)
    print(f"Fold {fold}: acc={m['acc']:.3f} prec={m['prec']:.3f} "
          f"rec={m['rec']:.3f} f1={m['f1']:.3f}")

mean = {k: np.mean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
std  = {k: np.std ([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
print("\nCV mean ± std:")
for k in mean:
    print(f"{k:>5}: {mean[k]:.3f} ± {std[k]:.3f}")

# ──────────────────────────────────────────────────────────────
# 5. Retrain on full 85 % dev  →  evaluate once on the 15 % test
# ──────────────────────────────────────────────────────────────
print("\n── Retrain on 85 % dev and evaluate on 15 % test ──")
final_metrics, final_model = train_one_split(X_dev, y_dev, X_dev, y_dev)
# evaluate on the unseen test set
test_dl = make_loader(X_test, y_test)
final_model.eval(); all_p, all_t = [], []
with torch.no_grad():
    for xb, yb in test_dl:
        prob = torch.sigmoid(final_model(xb.to(DEVICE))).cpu()
        all_p.append(prob); all_t.append(yb)
p = torch.cat(all_p).numpy(); t = torch.cat(all_t).numpy()
test_metrics = metrics_from_probs(p, t)

print("\nTest-set metrics:")
for k,v in test_metrics.items():
    print(f"{k:>5}: {v:.3f}")

# optional confusion matrix
cm = confusion_matrix(t, (p>0.5).astype(int))
plt.imshow(cm, cmap='Blues'); plt.title("15 % Hold-out Confusion Matrix")
plt.colorbar(); plt.xticks([0,1]); plt.yticks([0,1])
plt.xlabel("Pred"); plt.ylabel("True")
for i in range(2):
    for j in range(2):
        plt.text(j,i, cm[i,j],
                 ha='center', va='center',
                 color='white' if cm[i,j]>cm.max()/2 else 'black')
plt.show()
