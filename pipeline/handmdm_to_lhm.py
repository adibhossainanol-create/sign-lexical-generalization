"""
handmdm_to_lhm_v2.py
====================
Fixed conversion using HandMDM's own np_feats_to_smplx logic exactly,
including the negative sign on axis-angle conversion and lower body
joint insertion.
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

# ── Add HandMDM to path so we can use its geometry tools directly ──
HANDMDM_DIR = os.environ.get("HANDMDM_DIR", "../HandMDM")  # checkout providing src.tools.geometry
sys.path.insert(0, HANDMDM_DIR)

from src.tools.geometry import to_matrix, matrix_to

# Layout for 274-dim HandMDM output (confirmed from motion_feats_to_smpl.py)
SMPLX_KEYS_274 = {
    'body_pose':       (0,   78),   # 13 joints × 6 = 78
    'left_hand_pose':  (78,  168),  # 15 joints × 6 = 90
    'right_hand_pose': (168, 258),  # 15 joints × 6 = 90
    'jaw_pose':        (258, 264),  # 1 joint × 6 = 6
    'expression':      (264, 274),  # 10 dims
}

# Lower body joint indices that need zero-padding (HandMDM only has upper body)
BODY_POSE_LOWER_JOINTS = [0, 1, 3, 4, 6, 7, 9, 10]


def handmdm_6d_to_axis_angle(params_6d: np.ndarray, negate: bool = True) -> np.ndarray:
    """
    Convert 6D rotation to axis-angle exactly as HandMDM does:
        params = -matrix_to("axisangle", to_matrix("rot6d", params))
    
    Args:
        params_6d: (N_joints, 6) float array
        negate: apply the negative sign HandMDM uses (default True)
    Returns:
        (N_joints, 3) axis-angle array
    """
    params_torch = torch.tensor(params_6d, dtype=torch.float32)
    rot_mat = to_matrix("rot6d", params_torch)          # (N, 3, 3)
    axis_angle = matrix_to("axisangle", rot_mat)        # (N, 3)
    if negate:
        axis_angle = -axis_angle
    return axis_angle.detach().numpy()


def insert_lower_body_zeros(body_pose_aa: np.ndarray) -> np.ndarray:
    """
    HandMDM outputs 13 upper-body joints.
    LHM/SMPL-X needs 21 joints.
    Insert zero rotations at lower body joint indices.
    
    Args:
        body_pose_aa: (13, 3) upper body joints
    Returns:
        (21, 3) full body joints with zeros at lower body positions
    """
    result = body_pose_aa.copy()
    for idx in BODY_POSE_LOWER_JOINTS:
        result = np.insert(result, idx, [0.0, 0.0, 0.0], axis=0)
    return result  # (21, 3)


def convert_frame(frame: np.ndarray) -> dict:
    """
    Convert one frame of HandMDM 274-dim output to SMPL-X parameter dict,
    using HandMDM's exact conversion logic.
    
    Args:
        frame: (274,) float array
    Returns:
        dict with keys: body_pose, left_hand_pose, right_hand_pose, jaw_pose, expression
    """
    result = {}

    # Extract full body block (13 joints including root)
    body_start, body_end = SMPLX_KEYS_274['body_pose']
    body_6d = frame[body_start:body_end].reshape(13, 6)
    body_aa = handmdm_6d_to_axis_angle(body_6d, negate=True)  # (13, 3)

    # Extract root pose from joint 0
    # SMPL-X front-facing = pi rotation on X axis
    # HandMDM adds a small lean on top of that
    import math
    result['root_pose'] = np.array([math.pi, 0.0, 0.0])

    # All 13 joints go into body_pose (matching HandMDM own logic)
    # Insert 8 lower body zeros → 13 + 8 = 21 joints total
    full_body = body_aa.copy()
    for idx in BODY_POSE_LOWER_JOINTS:
        full_body = np.insert(full_body, idx, [0.0, 0.0, 0.0], axis=0)
    result['body_pose'] = full_body  # (21, 3)

    for key in ['left_hand_pose', 'right_hand_pose', 'jaw_pose']:
        start, end = SMPLX_KEYS_274[key]
        params = frame[start:end].reshape(-1, 6)
        result[key] = handmdm_6d_to_axis_angle(params, negate=True)

    # Expression — not a rotation, just copy raw values
    start, end = SMPLX_KEYS_274['expression']
    result['expression'] = frame[start:end].reshape(1, -1)

    return result


def load_reference_smplx(json_path: str) -> dict:
    """Load reference SMPL-X JSON from LHM's pose estimator."""
    with open(json_path) as f:
        d = json.load(f)
    return {
        'betas':       np.array(d['betas']),
        'root_pose':   np.array(d['root_pose']),
        'body_pose':   np.array(d['body_pose']),
        'jaw_pose':    np.array(d['jaw_pose']),
        'leye_pose':   np.array(d['leye_pose']),
        'reye_pose':   np.array(d['reye_pose']),
        'lhand_pose':  np.array(d['lhand_pose']),
        'rhand_pose':  np.array(d['rhand_pose']),
        'trans':       np.array(d['trans']),
        'focal':       np.array(d['focal']),
        'princpt':     np.array(d['princpt']),
        'img_size_wh': np.array(d['img_size_wh']),
        'pad_ratio':   float(d['pad_ratio']),
    }


def merge_frame(ref: dict, hm: dict, betas=None) -> dict:
    """
    Merge HandMDM motion with image-derived SMPL-X params into
    LHM's exact JSON schema per frame.

    Sources:
    - betas → the SIGNER's own shape when supplied, else the reference
      file's. NOTE the reference default gives every identity the same
      body, which silently undercuts the personalisation claim; LHM
      already estimates per-person betas and dumps them to
      LHM/exps/betas/<signer>.json, so pass those in.
    - root_pose, trans, camera → from the reference (stable framing)
    - body_pose → HandMDM (upper body motion drives realism)
    - lhand_pose, rhand_pose → HandMDM (sign language hand motion)
    - jaw/eye poses → neutral zeros
    """
    return {
        'betas':       (list(betas) if betas is not None
                        else ref['betas'].tolist()),
        'root_pose':   hm['root_pose'].tolist(),         # HandMDM root (fixes tilt)
        'body_pose':   hm['body_pose'].tolist(),        # (21,3) from HandMDM
        'jaw_pose':    hm['jaw_pose'].flatten().tolist()[:3],  # (3,)
        'leye_pose':   [0.0, 0.0, 0.0],
        'reye_pose':   [0.0, 0.0, 0.0],
        'lhand_pose':  hm['left_hand_pose'].tolist(),   # (15,3) from HandMDM
        'rhand_pose':  hm['right_hand_pose'].tolist(),  # (15,3) from HandMDM
        'trans':       [ref['trans'][0], ref['trans'][1], ref['trans'][2] * 0.6],  # closer to camera
        'focal':       ref['focal'].tolist(),
        'princpt':     ref['princpt'].tolist(),
        'img_size_wh': ref['img_size_wh'].tolist(),
        'pad_ratio':   ref['pad_ratio'],
    }


def run_conversion(handmdm_npy: str, ref_smplx_json: str, output_dir: str, betas=None):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[1/4] Loading HandMDM output: {handmdm_npy}")
    data = np.load(handmdm_npy, allow_pickle=True)
    if data.dtype == object:
        data = data.item()
    print(f"      shape: {data.shape}, dtype: {data.dtype}")

    print(f"[2/4] Loading reference SMPL-X: {ref_smplx_json}")
    ref = load_reference_smplx(ref_smplx_json)

    n_frames = data.shape[0]
    print(f"[3/4] Converting {n_frames} frames using HandMDM's exact conversion...")

    for i in range(n_frames):
        hm = convert_frame(data[i])
        frame_dict = merge_frame(ref, hm, betas=betas)
        out_path = os.path.join(output_dir, f"{i:05d}.json")
        with open(out_path, 'w') as f:
            json.dump(frame_dict, f, indent=2)

    print(f"[4/4] Done. Written {n_frames} JSON files to {output_dir}")
    print(f"\nRun LHM with:")
    print(f"  bash inference.sh LHM-500M-HF <image_folder> {output_dir}")


def validate_output(output_dir: str, ref_json: str):
    import glob
    files = sorted(glob.glob(os.path.join(output_dir, "*.json")))
    if not files:
        print("No output files to validate.")
        return

    with open(files[0]) as f:
        out = json.load(f)
    with open(ref_json) as f:
        ref = json.load(f)

    print("\n── Validation ──")
    print(f"{'Key':<15} {'Expected':>15} {'Got':>15} {'Match':>8}")
    print("─" * 55)
    all_ok = True
    for k in ref:
        ref_arr = np.array(ref[k])
        if k not in out:
            print(f"{k:<15} {str(ref_arr.shape):>15} {'MISSING':>15} {'✗':>8}")
            all_ok = False
            continue
        out_arr = np.array(out[k])
        match = ref_arr.shape == out_arr.shape
        if not match:
            all_ok = False
        print(f"{k:<15} {str(ref_arr.shape):>15} {str(out_arr.shape):>15} {'✓' if match else '✗':>8}")
    print("─" * 55)
    print("All good!" if all_ok else "Schema mismatch — check ✗ rows above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--handmdm_output", required=True)
    parser.add_argument("--ref_smplx", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--betas", default=None,
                        help="JSON file with this signer's 10 SMPL-X betas "
                             "(LHM/exps/betas/<signer>.json). Without it every "
                             "identity inherits the reference person's body.")
    args = parser.parse_args()

    _betas = None
    if args.betas:
        _betas = json.load(open(args.betas))
        if isinstance(_betas, dict):
            _betas = _betas.get("betas", _betas)
        print(f"      using signer betas from {args.betas}: "
              f"{[round(float(v),3) for v in _betas[:4]]}...")
    run_conversion(args.handmdm_output, args.ref_smplx, args.output_dir, betas=_betas)

    if args.validate:
        validate_output(args.output_dir, args.ref_smplx)
