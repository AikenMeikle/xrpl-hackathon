"""DirectedEdgeGraphSAGE: a small directed, edge-aware message-passing network
with an edge-level classifier, in native PyTorch (no PyG / torch_scatter).

* Node input = standardised wallet profile (activity, USD, degrees, time and
  asset diversity, pass-through, hub-likeness, neighbour stats).
* Edge input = standardised transaction, matching and motif features.
* Each layer: incoming-direction messages (from source node + edge attrs to
  destination) and outgoing-direction messages (from destination + edge attrs
  to source) are aggregated separately with degree normalisation, combined with
  the node's own state, residual connection, LayerNorm, GELU, dropout.
* Edge head: MLP over concat(h_src, h_dst, |h_src-h_dst|, h_src*h_dst,
  enc(edge), motif features) -> 128 -> 64 -> 1.
* No learned wallet-id embeddings: encoders are shared so the model transfers
  to the independent test graph. Transactions that are not economic edges get
  zero node context and are still scored by the edge head.
* Scalers are fitted on the training fold only. Early stopping on held-out AP.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from src.config import GNN_PARAMS, SEED
from src.features_wallet import NB_COLS, PROJECT_COLS

NODE_COLS = PROJECT_COLS + ["component_size", "community_size", "core", "clustering", "pagerank"] + [f"nb1_{c}" for c in NB_COLS]
# Full-batch training = one optimiser step per epoch, so more epochs and a larger
# learning rate than a mini-batch recipe. Early stopping on held-out AP still applies.
TRAIN_DEFAULTS = dict(lr=3e-3, epochs=150, patience=20)
MOTIF_PREFIXES = ("match_", "fanin_", "fanout_", "sg_", "gs_", "cycle_", "chain_", "bip_", "irr_")
EDGE_PREFIXES = ("tx_", "asof_", "win_", "fwd_", "full_pair", "full_diff", "full_same")


def gpu_report() -> dict:
    rep = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        try:
            rep["device_name"] = torch.cuda.get_device_name(0)
            rep["capability"] = torch.cuda.get_device_capability(0)
            rep["arch_list"] = torch.cuda.get_arch_list()
            x = torch.randn(256, 256, device="cuda")
            rep["smoke_matmul_ok"] = bool(torch.isfinite((x @ x).sum()).item())
            rep["total_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception as exc:  # unsupported arch etc. -> CPU
            rep["error"] = repr(exc)
            rep["cuda_available"] = False
    return rep


def pick_device() -> torch.device:
    rep = gpu_report()
    if rep.get("cuda_available") and rep.get("smoke_matmul_ok"):
        return torch.device("cuda")
    return torch.device("cpu")


class Standardiser:
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit(self, x: np.ndarray) -> "Standardiser":
        x = np.asarray(x, dtype=np.float64)
        self.mu = np.nanmean(x, axis=0)
        self.sd = np.nanstd(x, axis=0) + 1e-6
        self.mu = np.nan_to_num(self.mu)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.mu) / self.sd
        return np.clip(np.nan_to_num(z), -8, 8).astype(np.float32)


def _signed_log(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


class DirectedSAGELayer(nn.Module):
    def __init__(self, h: int, e: int, dropout: float):
        super().__init__()
        self.msg_in = nn.Linear(2 * h + e, h)    # message to destination (incoming direction)
        self.msg_out = nn.Linear(2 * h + e, h)   # message to source (outgoing direction)
        self.combine = nn.Linear(3 * h, h)
        self.norm = nn.LayerNorm(h)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, ea: torch.Tensor,
                in_deg: torch.Tensor, out_deg: torch.Tensor) -> torch.Tensor:
        n, h = x.shape
        pair = torch.cat([x[src], x[dst], ea], dim=1)
        m_in = F.gelu(self.msg_in(pair))
        m_out = F.gelu(self.msg_out(pair))
        agg_in = torch.zeros(n, h, device=x.device, dtype=x.dtype).index_add_(0, dst, m_in) / in_deg
        agg_out = torch.zeros(n, h, device=x.device, dtype=x.dtype).index_add_(0, src, m_out) / out_deg
        out = self.combine(torch.cat([x, agg_in, agg_out], dim=1))
        out = self.norm(x + self.drop(F.gelu(out)))
        return out


class DirectedEdgeGraphSAGE(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, motif_dim: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.node_enc = nn.Sequential(nn.Linear(node_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.edge_enc = nn.Sequential(nn.Linear(edge_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.layers = nn.ModuleList([DirectedSAGELayer(hidden, hidden, dropout) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.Linear(4 * hidden + hidden + motif_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, nodes: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, edge_attr: torch.Tensor,
                motif: torch.Tensor, has_graph: torch.Tensor) -> torch.Tensor:
        n = nodes.shape[0]
        x = self.node_enc(nodes)
        ea = self.edge_enc(edge_attr)
        valid = has_graph > 0.5
        s, d, ea_v = src[valid], dst[valid], ea[valid]
        in_deg = torch.ones(n, 1, device=x.device).index_add_(0, d, torch.ones(len(d), 1, device=x.device))
        out_deg = torch.ones(n, 1, device=x.device).index_add_(0, s, torch.ones(len(s), 1, device=x.device))
        for layer in self.layers:
            x = layer(x, s, d, ea_v, in_deg, out_deg)
        hs = torch.where(valid[:, None], x[src.clamp(min=0)], torch.zeros_like(ea))
        hd = torch.where(valid[:, None], x[dst.clamp(min=0)], torch.zeros_like(ea))
        z = torch.cat([hs, hd, (hs - hd).abs(), hs * hd, ea, motif], dim=1)
        return self.head(z).squeeze(1)


# --------------------------------------------------------------------------
# data assembly
# --------------------------------------------------------------------------
def assemble(feats: pd.DataFrame, edges: pd.DataFrame, prof: pd.DataFrame) -> dict:
    """Tensors for one split. Rows follow ``feats`` order (canonical order)."""
    wallets = pd.Index(pd.unique(np.concatenate([edges["source_wallet"].to_numpy(), edges["destination_wallet"].to_numpy()])))
    wmap = {w: i for i, w in enumerate(wallets)}
    e_idx = edges.set_index("transaction_hash")
    src = feats["transaction_hash"].map(e_idx["source_wallet"]).map(wmap)
    dst = feats["transaction_hash"].map(e_idx["destination_wallet"]).map(wmap)
    has_graph = src.notna() & dst.notna()
    node_cols = [c for c in NODE_COLS if c in prof.columns]
    node_raw = _signed_log(prof.reindex(range(len(wallets)))[node_cols].to_numpy(dtype=float))
    edge_cols = [c for c in feats.columns if c.startswith(EDGE_PREFIXES) and not c.startswith("full_src") and not c.startswith("full_dst")]
    motif_cols = [c for c in feats.columns if c.startswith(MOTIF_PREFIXES)]
    return {
        "src": src.fillna(-1).to_numpy(dtype=np.int64), "dst": dst.fillna(-1).to_numpy(dtype=np.int64),
        "has_graph": has_graph.to_numpy(dtype=np.float32), "node_raw": node_raw,
        "edge_raw": _signed_log(feats[edge_cols].to_numpy(dtype=float)),
        "motif_raw": _signed_log(feats[motif_cols].to_numpy(dtype=float)),
        "edge_cols": edge_cols, "motif_cols": motif_cols, "node_cols": node_cols, "n_nodes": len(wallets),
    }


def _to_tensors(data: dict, sc_node: Standardiser, sc_edge: Standardiser, sc_motif: Standardiser, device) -> dict:
    return {
        "nodes": torch.tensor(sc_node.transform(data["node_raw"]), device=device),
        "src": torch.tensor(data["src"], device=device), "dst": torch.tensor(data["dst"], device=device),
        "edge_attr": torch.tensor(sc_edge.transform(data["edge_raw"]), device=device),
        "motif": torch.tensor(sc_motif.transform(data["motif_raw"]), device=device),
        "has_graph": torch.tensor(data["has_graph"], device=device),
    }


def _train(model, T, y, train_mask, val_mask, device, epochs, patience, lr, wd, log=print) -> tuple[int, float, list]:
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    tm = torch.tensor(train_mask, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_ap, best_epoch, best_state, bad = -1.0, 0, None, 0
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(T["nodes"], T["src"], T["dst"], T["edge_attr"], T["motif"], T["has_graph"])
        loss = F.binary_cross_entropy_with_logits(logits[tm], yt[tm])
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite GNN loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        loss_v = float(loss.detach())
        if val_mask is None:
            history.append({"epoch": ep, "loss": loss_v})
            continue
        model.eval()
        with torch.no_grad():
            s = torch.sigmoid(model(T["nodes"], T["src"], T["dst"], T["edge_attr"], T["motif"], T["has_graph"])).cpu().numpy()
        ap = float(average_precision_score(y[val_mask], s[val_mask]))
        history.append({"epoch": ep, "loss": loss_v, "val_ap": ap})
        if ap > best_ap + 1e-5:
            best_ap, best_epoch, bad = ap, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch, best_ap, history


def _scores(model, T) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(T["nodes"], T["src"], T["dst"], T["edge_attr"], T["motif"], T["has_graph"])).cpu().numpy()


def oof_gnn(feats: pd.DataFrame, edges: pd.DataFrame, prof: pd.DataFrame, y: np.ndarray, folds, params: dict | None = None) -> dict:
    """Out-of-fold GNN scores on the same blocked folds as LightGBM.

    Message passing runs over the full (unlabelled) training graph; the loss and
    the early-stopping metric only ever see training-fold / validation-fold
    labels respectively, and scalers are fitted on training-fold rows only.
    Validation-fold edges therefore contribute structure but never labels.
    """
    p = {**GNN_PARAMS, **TRAIN_DEFAULTS, **(params or {})}
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = pick_device()
    data = assemble(feats, edges, prof)
    oof = np.full(len(y), np.nan)
    fold_ap, best_epochs, n_params = [], [], 0
    t0 = time.time()
    for k, (tr, va) in enumerate(folds, 1):
        tr_mask = np.zeros(len(y), dtype=bool); tr_mask[tr] = True
        va_mask = np.zeros(len(y), dtype=bool); va_mask[va] = True
        # node scaler: wallets touched by training-fold edges only
        tr_nodes = np.unique(np.concatenate([data["src"][tr], data["dst"][tr]]))
        tr_nodes = tr_nodes[tr_nodes >= 0]
        sc_node = Standardiser().fit(data["node_raw"][tr_nodes])
        sc_edge = Standardiser().fit(data["edge_raw"][tr])
        sc_motif = Standardiser().fit(data["motif_raw"][tr])
        T = _to_tensors(data, sc_node, sc_edge, sc_motif, device)
        torch.manual_seed(SEED + k)
        model = DirectedEdgeGraphSAGE(len(data["node_cols"]), len(data["edge_cols"]), len(data["motif_cols"]),
                                      p["hidden"], p["layers"], p["dropout"]).to(device)
        n_params = sum(q.numel() for q in model.parameters())
        best_epoch, best_ap, _hist = _train(model, T, y, tr_mask, va_mask, device, p["epochs"], p["patience"], p["lr"], p["weight_decay"])
        s = _scores(model, T)
        oof[va] = s[va]
        fold_ap.append(float(average_precision_score(y[va], s[va])))
        best_epochs.append(best_epoch)
        print(f"    GNN fold {k}: best_epoch={best_epoch} AP={100 * fold_ap[-1]:.2f} ({time.time() - t0:.0f}s)", flush=True)
    if np.isnan(oof).any():
        raise AssertionError("GNN OOF has unscored rows")
    return {"oof": oof, "fold_ap": fold_ap, "mean_ap": float(np.mean(fold_ap)), "best_epochs": best_epochs,
            "device": str(device), "n_params": int(n_params), "gpu": gpu_report()}


def fit_final_gnn(feats, edges, prof, y, epochs: int, out_path: Path, params: dict | None = None) -> dict:
    p = {**GNN_PARAMS, **TRAIN_DEFAULTS, **(params or {})}
    torch.manual_seed(SEED)
    device = pick_device()
    data = assemble(feats, edges, prof)
    sc_node = Standardiser().fit(data["node_raw"])
    sc_edge = Standardiser().fit(data["edge_raw"])
    sc_motif = Standardiser().fit(data["motif_raw"])
    T = _to_tensors(data, sc_node, sc_edge, sc_motif, device)
    model = DirectedEdgeGraphSAGE(len(data["node_cols"]), len(data["edge_cols"]), len(data["motif_cols"]),
                                  p["hidden"], p["layers"], p["dropout"]).to(device)
    _train(model, T, y, np.ones(len(y), dtype=bool), None, device, max(epochs, 1), p["patience"], p["lr"], p["weight_decay"])
    torch.save({"state": model.state_dict(), "node_cols": data["node_cols"], "edge_cols": data["edge_cols"],
                "motif_cols": data["motif_cols"], "scalers": {"node": (sc_node.mu, sc_node.sd), "edge": (sc_edge.mu, sc_edge.sd),
                                                               "motif": (sc_motif.mu, sc_motif.sd)}, "params": p, "epochs": epochs}, out_path)
    return {"device": str(device), "epochs": epochs, "n_params": int(sum(q.numel() for q in model.parameters()))}


def predict_gnn(feats, edges, prof, ckpt_path: Path) -> np.ndarray:
    ck = torch.load(ckpt_path, weights_only=False)
    device = pick_device()
    data = assemble(feats, edges, prof)
    if data["edge_cols"] != ck["edge_cols"] or data["motif_cols"] != ck["motif_cols"]:
        raise AssertionError("test feature columns differ from the trained GNN")
    scs = []
    for key in ("node", "edge", "motif"):
        s = Standardiser(); s.mu, s.sd = ck["scalers"][key]; scs.append(s)
    T = _to_tensors(data, *scs, device)
    p = ck["params"]
    model = DirectedEdgeGraphSAGE(len(ck["node_cols"]), len(ck["edge_cols"]), len(ck["motif_cols"]), p["hidden"], p["layers"], p["dropout"]).to(device)
    model.load_state_dict(ck["state"])
    return _scores(model, T)
