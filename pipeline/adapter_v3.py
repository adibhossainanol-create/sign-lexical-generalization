
import argparse

import numpy as np
import torch
import torch.nn as nn

from retarget_loss import RetargetLoss, SMPLXHands, MODEL_PATH_DEFAULT


# ---- identical to v2 -------------------------------------------------------
def degrade(seq, rng, compress=0.35, smooth_k=9, static_p=0.25):
    F = seq.shape[0]
    mean_pose = seq.mean(axis=0, keepdims=True)
    deg = mean_pose + (seq - mean_pose) * compress
    if F >= smooth_k:
        kernel = np.ones(smooth_k, dtype=np.float32) / smooth_k
        deg = np.stack([np.convolve(deg[:, d], kernel, mode="same")
                        for d in range(seq.shape[1])], axis=1)
    if rng.random() < static_p and F > 6:
        s = rng.integers(0, F - 4)
        L = rng.integers(3, min(F - s, F // 2 + 1))
        deg[s:s + L] = deg[s]
    return deg.astype(np.float32)


class Adapter(nn.Module):
    def __init__(self, dim=90, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, dim, 5, padding=2),
        )

    def forward(self, x):
        return x + self.net(x)
# ---------------------------------------------------------------------------


def make_windows(clips, idx, rng, gt_lookup, n_shapes, win=32):
    """Windows of degraded input + the cached retargeted GT for the matching
    (clip, shape) pair. Also returns per-window valid length so padded tail
    frames can be masked out of the loss."""
    X, GT, SH, VL = [], [], [], []
    for i in idx:
        clean = clips[i].astype(np.float32)
        if clean.shape[0] < 8:
            continue
        deg = degrade(clean, rng)
        si = int(rng.integers(0, n_shapes))       # one body shape per clip
        gt_j = gt_lookup[(int(i), si)]            # (F, 30, 3)
        F = clean.shape[0]
        for s in range(0, max(1, F - win + 1), win // 2):
            d = deg[s:s + win]
            g = gt_j[s:s + win]
            valid = d.shape[0]
            if valid < win:
                d = np.pad(d, ((0, win - valid), (0, 0)))
                g = np.pad(g, ((0, win - valid), (0, 0), (0, 0)))
            X.append(d); GT.append(g); SH.append(si); VL.append(valid)
    X = np.array(X, dtype=np.float32).transpose(0, 2, 1)   # (N, 90, T)
    return (torch.tensor(X), torch.tensor(np.array(GT, dtype=np.float32)),
            torch.tensor(np.array(SH)), torch.tensor(np.array(VL)))


def masked_pos_mm(pred_pose, gt_j, betas, valid, fk, chunk_windows=16):
    """Held-out metric: mean 3D hand-joint error in mm over valid frames."""
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for s in range(0, pred_pose.shape[0], chunk_windows):
            p = pred_pose[s:s + chunk_windows]
            g = gt_j[s:s + chunk_windows]
            b = betas[s:s + chunk_windows]
            v = valid[s:s + chunk_windows]
            B, F, _ = p.shape
            j = fk(p.reshape(B * F, 90),
                   b.unsqueeze(1).expand(B, F, 10).reshape(B * F, 10)
                   ).reshape(B, F, 30, 3)
            err = (j - g).norm(dim=-1)                       # (B, F, 30)
            m = (torch.arange(F)[None, :] < v[:, None]).float().unsqueeze(-1)
            tot += (err * m).sum().item()
            cnt += m.sum().item() * 30
    return 1000.0 * tot / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--win", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-vel", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--model-path", default=MODEL_PATH_DEFAULT)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    clips, train_idx, test_idx = d["clips"], d["train_idx"], d["test_idx"]

    c = np.load(args.gt, allow_pickle=True)
    joints, clip_id, shape_id, shapes = (c["joints"], c["clip_id"],
                                         c["shape_id"], c["shapes"])
    gt_lookup = {(int(clip_id[k]), int(shape_id[k])): joints[k]
                 for k in range(len(clip_id))}
    n_shapes = shapes.shape[0]
    shapes_t = torch.tensor(shapes, dtype=torch.float32)
    print(f"loaded GT cache: {len(gt_lookup)} entries, {n_shapes} shapes")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    Xtr, Gtr, Str, Vtr = make_windows(clips, train_idx, rng, gt_lookup, n_shapes, args.win)
    Xte, Gte, Ste, Vte = make_windows(clips, test_idx, rng, gt_lookup, n_shapes, args.win)
    print(f"train windows {Xtr.shape[0]}, held-out {Xte.shape[0]}")

    fk = SMPLXHands(args.model_path)
    Bte = shapes_t[Ste]

    base = masked_pos_mm(Xte.transpose(1, 2), Gte, Bte, Vte, fk)
    print(f"no-op baseline position error (held-out): {base:.2f} mm")

    model = Adapter()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = RetargetLoss(args.model_path, w_vel=args.w_vel)

    N = Xtr.shape[0]
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(N)
        run = 0.0
        for s in range(0, N, args.bs):
            b = perm[s:s + args.bs]
            opt.zero_grad()
            out = model(Xtr[b]).transpose(1, 2)          # (B, T, 90)
            loss, parts = lossf(out, Gtr[b], shapes_t[Str[b]])
            loss.backward(); opt.step()
            run += loss.item()
        model.eval()
        te = masked_pos_mm(model(Xte).transpose(1, 2), Gte, Bte, Vte, fk)
        print(f"  epoch {ep+1:3d}  train {run/max(1,N//args.bs):.5f}  "
              f"held-out {te:.2f} mm", flush=True)

    model.eval()
    final = masked_pos_mm(model(Xte).transpose(1, 2), Gte, Bte, Vte, fk)
    imp = 100 * (base - final) / base
    print("\n=== RESULT (retargeting loss) ===")
    print(f"no-op baseline : {base:.2f} mm")
    print(f"adapter        : {final:.2f} mm")
    print(f"improvement    : {imp:+.1f}%   (w_vel={args.w_vel})")
    print("NOTE: mm, not comparable to v2's radian MPJAE figure.")
    torch.save(model.state_dict(), "adapter_v3.pt")


if __name__ == "__main__":
    main()
