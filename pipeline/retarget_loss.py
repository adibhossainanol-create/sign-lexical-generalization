
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

MODEL_PATH_DEFAULT = os.environ.get(
    "SMPLX_MODEL_PATH",
    os.path.join(os.environ.get("LHM_DIR", "."), "pretrained_models/human_model_files"))

# SMPL-X joint indices
J_LWRIST, J_RWRIST = 20, 21
LHAND = list(range(25, 40))   # 15 left finger joints
RHAND = list(range(40, 55))   # 15 right finger joints


class SMPLXHands(nn.Module):
    """Differentiable map: (hand_pose_90, betas) -> wrist-relative 3D hand joints.

    smplx.create fixes batch_size at construction, so models are cached per
    size and inputs are processed in chunks.
    """

    def __init__(self, model_path=MODEL_PATH_DEFAULT, chunk=256):
        super().__init__()
        import smplx
        self._smplx = smplx
        self.model_path = model_path
        self.chunk = chunk
        self._cache = {}

    def _model(self, bs):
        if bs not in self._cache:
            self._cache[bs] = self._smplx.create(
                self.model_path, model_type="smplx", gender="neutral", ext="npz",
                use_pca=False, flat_hand_mean=True, batch_size=bs,
            )
        return self._cache[bs]

    def forward(self, pose90, betas):
        """pose90: (N, 90) axis-angle, 45 left then 45 right.
        betas:  (N, 10).  Returns (N, 30, 3), wrist-relative."""
        outs = []
        for i in range(0, pose90.shape[0], self.chunk):
            p = pose90[i:i + self.chunk]
            b = betas[i:i + self.chunk]
            n = p.shape[0]
            m = self._model(n)
            res = m(betas=b,
                    left_hand_pose=p[:, :45].reshape(n, 15, 3),
                    right_hand_pose=p[:, 45:].reshape(n, 15, 3),
                    return_verts=False)
            J = res.joints
            L = J[:, LHAND] - J[:, J_LWRIST].unsqueeze(1)
            R = J[:, RHAND] - J[:, J_RWRIST].unsqueeze(1)
            outs.append(torch.cat([L, R], dim=1))
        return torch.cat(outs, dim=0)


class RetargetLoss(nn.Module):
    """Position loss on retargeted 3D hand joints, plus a velocity term.

    pred_pose: (B, F, 90) adapter output, axis-angle
    gt_joints: (B, F, 30, 3) precomputed retargeted ground truth
    betas:     (B, 10) body shape -- MUST be the same shape used to build gt_joints
    """

    def __init__(self, model_path=MODEL_PATH_DEFAULT, w_vel=1.0, chunk=256):
        super().__init__()
        self.fk = SMPLXHands(model_path, chunk=chunk)
        self.w_vel = w_vel

    def forward(self, pred_pose, gt_joints, betas):
        B, F, _ = pred_pose.shape
        flat_pose = pred_pose.reshape(B * F, 90)
        flat_betas = betas.unsqueeze(1).expand(B, F, 10).reshape(B * F, 10)

        pred_j = self.fk(flat_pose, flat_betas).reshape(B, F, 30, 3)

        pos = (pred_j - gt_joints).norm(dim=-1).mean()

        if F > 1 and self.w_vel > 0:
            dv_p = pred_j[:, 1:] - pred_j[:, :-1]
            dv_g = gt_joints[:, 1:] - gt_joints[:, :-1]
            vel = (dv_p - dv_g).norm(dim=-1).mean()
        else:
            vel = torch.zeros((), device=pred_pose.device)

        return pos + self.w_vel * vel, {"pos_mm": pos.item() * 1000,
                                        "vel_mm": vel.item() * 1000}


def precompute(npz_path, out_path, n_shapes=8, seed=0,
               model_path=MODEL_PATH_DEFAULT, chunk=256):
    """Cache retargeted GT hand joints for every clip x a fixed pool of shapes.

    A fixed pool (rather than fresh random shapes per batch) keeps the target
    deterministic, makes runs comparable, and halves per-epoch cost since the
    GT side never re-runs SMPL-X.
    """
    d = np.load(npz_path, allow_pickle=True)
    clips = list(d["clips"])
    rng = np.random.RandomState(seed)
    shapes = rng.randn(n_shapes, 10).astype(np.float32) * 0.5
    shapes[0] = 0.0                      # keep neutral in the pool

    fk = SMPLXHands(model_path, chunk=chunk)
    all_j, clip_id, shape_id = [], [], []

    with torch.no_grad():
        for ci, c in enumerate(clips):
            p = torch.from_numpy(np.asarray(c, dtype=np.float32))   # (F,90)
            F = p.shape[0]
            for si in range(n_shapes):
                b = torch.from_numpy(shapes[si]).unsqueeze(0).expand(F, 10)
                all_j.append(fk(p, b).numpy().astype(np.float32))
                clip_id.append(ci)
                shape_id.append(si)
            if ci % 50 == 0:
                print(f"  clip {ci}/{len(clips)}", flush=True)

    np.savez_compressed(out_path,
                        joints=np.array(all_j, dtype=object),
                        clip_id=np.array(clip_id),
                        shape_id=np.array(shape_id),
                        shapes=shapes)
    print(f"wrote {out_path}: {len(all_j)} (clip,shape) entries, "
          f"{n_shapes} shapes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["precompute", "selftest"])
    ap.add_argument("--npz")
    ap.add_argument("--out")
    ap.add_argument("--n-shapes", type=int, default=8)
    ap.add_argument("--model-path", default=MODEL_PATH_DEFAULT)
    a = ap.parse_args()

    if a.mode == "precompute":
        precompute(a.npz, a.out, a.n_shapes, model_path=a.model_path)
    else:
        # tiny end-to-end check: does the loss drop when pose moves toward GT?
        torch.manual_seed(0)
        fk = SMPLXHands(a.model_path)
        B, F = 2, 8
        gt_pose = torch.randn(B, F, 90) * 0.2
        betas = torch.randn(B, 10) * 0.5
        with torch.no_grad():
            flat_b = betas.unsqueeze(1).expand(B, F, 10).reshape(B * F, 10)
            gt_j = fk(gt_pose.reshape(B * F, 90), flat_b).reshape(B, F, 30, 3)

        lossf = RetargetLoss(a.model_path, w_vel=1.0)
        far = (gt_pose + torch.randn_like(gt_pose) * 0.3).requires_grad_(True)
        near = (gt_pose + torch.randn_like(gt_pose) * 0.05).requires_grad_(True)
        lf, df = lossf(far, gt_j, betas)
        ln, dn = lossf(near, gt_j, betas)
        lf.backward()
        print(f"far  loss {lf.item():.5f}  pos {df['pos_mm']:.1f}mm  vel {df['vel_mm']:.1f}mm")
        print(f"near loss {ln.item():.5f}  pos {dn['pos_mm']:.1f}mm  vel {dn['vel_mm']:.1f}mm")
        print("near < far:", ln.item() < lf.item())
        print("grad flows:", far.grad is not None and far.grad.abs().sum().item() > 0)
