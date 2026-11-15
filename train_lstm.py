"""LSTM, HORIZON weeks ahead, evaluated on the SAME test weeks as XGBoost.
Run after train_xgb.py: python train_lstm.py"""
import copy, json
import numpy as np
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data import load_data, time_split, WEATHER_COLS
from src.features import make_features, HORIZON

torch.manual_seed(42); np.random.seed(42)
SEQ = 12  # look back 12 weeks

df, synthetic = load_data()
df = df.reset_index(drop=True)
feat, _ = make_features(df)
train_f, test_f = time_split(feat)
train_f = train_f.iloc[:-HORIZON]

date2pos = {d: i for i, d in enumerate(df["week_start_date"])}
tr_pos = np.array([date2pos[d] for d in train_f["week_start_date"]])
te_pos = np.array([date2pos[d] for d in test_f["week_start_date"]])
tr_pos, te_pos = tr_pos[tr_pos >= SEQ - 1], te_pos[te_pos >= SEQ - 1]

cols = WEATHER_COLS + ["total_cases"]
sc = StandardScaler().fit(df.loc[: tr_pos.max() + HORIZON, cols])
arr = sc.transform(df[cols])


def windows(pos):
    X = np.stack([arr[p - SEQ + 1: p + 1] for p in pos])
    y = np.array([arr[p + HORIZON, -1] for p in pos])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


nval = max(int(len(tr_pos) * 0.15), 20)
Xtr, ytr = windows(tr_pos[:-nval])
Xva, yva = windows(tr_pos[-nval:])
Xte, yte = windows(te_pos)


class Net(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.lstm = nn.LSTM(n, 48, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(48, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(self.drop(o[:, -1])).squeeze(-1)


net = Net(len(cols))
opt = torch.optim.Adam(net.parameters(), lr=0.003, weight_decay=1e-4)
best, best_state, wait = 1e9, None, 0
for ep in range(150):
    net.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(perm), 32):
        b = perm[i:i + 32]
        opt.zero_grad()
        nn.functional.mse_loss(net(Xtr[b]), ytr[b]).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        v = nn.functional.mse_loss(net(Xva), yva).item()
    if v < best - 1e-4:
        best, best_state, wait = v, copy.deepcopy(net.state_dict()), 0
    else:
        wait += 1
    if ep % 10 == 0:
        print("epoch", ep, "val loss", round(v, 4))
    if wait >= 15:
        print("early stop at epoch", ep)
        break

net.load_state_dict(best_state); net.eval()
with torch.no_grad():
    p = net(Xte).numpy()
mu, sd = sc.mean_[-1], sc.scale_[-1]
pred = np.clip(p * sd + mu, 0, None)
actual = yte.numpy() * sd + mu

m = json.load(open("outputs/xgb_metrics.json"))
thr = m["outbreak_threshold"]
m["lstm"] = {"MAE": float(mean_absolute_error(actual, pred)),
             "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
             "outbreak_recall": float(((pred > thr) & (actual > thr)).sum() / max((actual > thr).sum(), 1))}
json.dump(m, open("outputs/xgb_metrics.json", "w"), indent=2)
print(f"\nCOMPARISON - {HORIZON} weeks ahead (test set, lower MAE/RMSE is better)")
for k in ("naive", "xgboost", "lstm"):
    print(f"{k:8s} MAE={m[k]['MAE']:.2f}  RMSE={m[k]['RMSE']:.2f}  outbreak_recall={m[k].get('outbreak_recall', float('nan')):.2f}")