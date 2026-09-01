#!/usr/bin/env python3
"""
CGReplay post-processing — video quality + QoE metrics.

Run from CGReplay/ root:
    python3 tools/post_process.py --mode quic
    python3 tools/post_process.py --mode rtp
    python3 tools/post_process.py --mode scream

Outputs: player/logs/metrics_{mode}.csv
Columns: frame_id, SSIM, PSNR, VMAF, fps, response_time_ms, QoE
Per-frame on a fixed 1..STOP_FRAME-1 axis; un-received frames are NaN (gaps).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import glob
import cv2
import pandas as pd
import yaml
from pyzbar import pyzbar

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from video_Quality import compare_images, mask_qr_with_ref, FFMPEG
from qoe import perceived_qoe

CONFIG_PATH   = "config/config.yaml"
LOGS_DIR      = os.path.join("player", "logs")

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

STOP_FRAME  = cfg["Running"]["stop_frm_number"]
FPS_TARGET  = cfg["encoding"]["fps"]
# Reference frames follow the game picked in config (Kombat / Forza / Fortnite),
# so the metrics compare against the dataset that was actually streamed.
REF_FOLDER  = os.path.join("server", cfg["Running"]["game"])


def _pick(*candidates: str) -> str | None:
    """Return the first path that exists (per-mode backup preferred)."""
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_fps_rt(mode: str) -> pd.DataFrame:
    """Return DataFrame with frame_id, fps, response_time_ms, timestamp columns.

    Prefers the per-mode log backups written by the topology
    (frame_{mode}.csv, responsetime_{mode}.csv) so a manual re-run or a later
    test doesn't read another mode's shared logs. Falls back to the shared
    files when a backup is absent.

    All three modes use responsetime for command latency, computed identically:
    time from frame receive to joystick command send.
    """
    rt_path = _pick(
        os.path.join(LOGS_DIR, f"responsetime_{mode}.csv"),
        os.path.join(LOGS_DIR, "responsetime_CG.csv"),
    )

    if mode in ("quic", "roq"):
        # QUIC and RoQ save a per-frame log with a recv_time column.
        _shared = "ply_quic_frame.csv" if mode == "quic" else "ply_roq_frame.csv"
        frame_path = _pick(
            os.path.join(LOGS_DIR, f"frame_{mode}.csv"),
            os.path.join(LOGS_DIR, _shared),
        )
    else:
        frame_path = _pick(
            os.path.join(LOGS_DIR, f"frame_{mode}.csv"),
            os.path.join(LOGS_DIR, "ply_frame.csv"),
        )

    if frame_path is None:
        # No fps log for this mode (shared file overwritten / not run yet).
        # Return an empty frame so SSIM/PSNR still compute; fps/RT stay NaN.
        print(f"  [warn] no frame log for mode={mode}; fps/RT will be NaN")
        return pd.DataFrame(columns=["frame_id", "fps", "response_time_ms"])

    if mode in ("quic", "roq"):
        df = pd.read_csv(frame_path)[["frame_id", "fps", "recv_time"]]
        df = df.rename(columns={"recv_time": "timestamp"})
    else:
        df = pd.read_csv(frame_path)[["frame_id", "fps"]]

    if rt_path and os.path.exists(rt_path):
        df_rt = pd.read_csv(rt_path)
        df_rt["response_time_ms"] = (
            (df_rt["cmd_timestamp"] - df_rt["frame_timestamp"]) * 1000.0
        )
        merge_cols = ["frame_id", "response_time_ms"]
        if "timestamp" not in df.columns:
            merge_cols += ["frame_timestamp"]
        df = df.merge(df_rt[merge_cols], on="frame_id", how="left")
        if "frame_timestamp" in df.columns:
            df = df.rename(columns={"frame_timestamp": "timestamp"})
    else:
        df["response_time_ms"] = float("nan")

    return df.sort_values("frame_id").reset_index(drop=True)


def _decode_frame_id(path: str) -> int | None:
    """Read the TRUE frame id from the QR burned into a received frame."""
    img = cv2.imread(path)
    if img is None:
        return None
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    for q in pyzbar.decode(blurred):
        for part in q.data.decode("utf-8", "ignore").split(","):
            if "Frame ID" in part:
                try:
                    return int(part.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass
    return None


def realign_by_qr(tgt_folder: str) -> str:
    """RTP/SCReAM save received frames by ARRIVAL COUNTER, not by true frame id,
    so any loss or reorder shifts every subsequent file out of alignment with the
    reference (e.g. saved 0011.png actually holds frame 26). Each received frame
    carries its true id in the burned-in QR; remap by decoding it into a temp dir
    of {true_id:04d}.png (last writer wins on duplicate retries). Falls back to the
    filename number when a QR can't be read. Without this, SSIM/PSNR/VMAF compare
    mismatched frames and collapse toward zero."""
    out = tempfile.mkdtemp(prefix="aligned_")
    n_qr = n_fallback = 0
    for p in sorted(glob.glob(os.path.join(tgt_folder, "*.png"))):
        fid = _decode_frame_id(p)
        if fid is not None:
            n_qr += 1
        else:
            try:
                fid = int(os.path.splitext(os.path.basename(p))[0])
            except ValueError:
                continue
            n_fallback += 1
        shutil.copy(p, os.path.join(out, f"{fid:04d}.png"))
    print(f"  Realigned by QR: {n_qr} frames by true id, {n_fallback} by filename fallback")
    return out


def compute_vmaf_sequence(tgt_folder: str, frame_ids: list[int]) -> dict:
    """Per-frame VMAF via a single libvmaf pass over the matched sequence.

    Builds two videos (reference + received) from the frames that were actually
    received, QR-masked the same way SSIM/PSNR are, then runs libvmaf once with
    JSON per-frame output. Returns {frame_id: vmaf}. One encode + one libvmaf
    per mode — much faster and more meaningful than per-frame (single-frame
    VMAF has no temporal context and scores far too low).
    """
    frame_ids = sorted(frame_ids)
    if not frame_ids:
        return {}

    # Unique per-run temp dir for ALL intermediates. Fixed /tmp paths broke when a
    # root (sudo Mininet) run left root-owned /tmp/vmaf_*.mp4/.json behind: a later
    # user run could not overwrite them, ffmpeg failed silently, and libvmaf read
    # the STALE json -> wrong (pre-realignment) VMAF. Unique paths avoid that.
    work     = tempfile.mkdtemp(prefix="vmaf_work_")
    ref_dir  = os.path.join(work, "ref");  os.makedirs(ref_dir)
    dist_dir = os.path.join(work, "dist"); os.makedirs(dist_dir)
    log_path = os.path.join(work, "vmaf_seq.json")
    index_to_fid = {}
    try:
        seq = 0
        for fid in frame_ids:
            ref  = cv2.imread(os.path.join(REF_FOLDER, f"{fid:04d}.png"))
            dist = cv2.imread(os.path.join(tgt_folder, f"{fid:04d}.png"))
            if ref is None or dist is None:
                continue
            if dist.shape != ref.shape:
                dist = cv2.resize(dist, (ref.shape[1], ref.shape[0]))
            dist_masked = mask_qr_with_ref(ref, dist)
            cv2.imwrite(os.path.join(ref_dir,  f"{seq:04d}.png"), ref)
            cv2.imwrite(os.path.join(dist_dir, f"{seq:04d}.png"), dist_masked)
            index_to_fid[seq] = fid
            seq += 1

        if seq == 0:
            return {}

        ref_mp4  = os.path.join(work, "vmaf_ref.mp4")
        dist_mp4 = os.path.join(work, "vmaf_dist.mp4")
        for src, out in [(ref_dir, ref_mp4), (dist_dir, dist_mp4)]:
            subprocess.run(
                [FFMPEG, "-y", "-framerate", str(FPS_TARGET),
                 "-i", os.path.join(src, "%04d.png"),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", out],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        subprocess.run(
            [FFMPEG, "-i", dist_mp4, "-i", ref_mp4, "-lavfi",
             f"[0:v][1:v]libvmaf=log_path={log_path}:log_fmt=json",
             "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        with open(log_path) as f:
            data = json.load(f)
        result = {}
        for fr in data.get("frames", []):
            idx = fr.get("frameNum", fr.get("frame"))
            vmaf = fr.get("metrics", {}).get("vmaf")
            if idx in index_to_fid and vmaf is not None:
                result[index_to_fid[idx]] = vmaf
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)


def compute_metrics(mode: str):
    print(f"[post_process] mode={mode}")

    quality_csv = os.path.join(LOGS_DIR, f"quality_{mode}.csv")
    metrics_csv = os.path.join(LOGS_DIR, f"metrics_{mode}.csv")

    # Use mode-specific backup if it exists, otherwise fall back to shared dir
    tgt_mode = os.path.join("player", f"received_frames_{mode}")
    tgt_fallback = os.path.join("player", "received_frames")
    tgt_folder = tgt_mode if os.path.isdir(tgt_mode) else tgt_fallback

    # RTP/SCReAM name received frames by arrival counter, so realign them to
    # their true ids (from the QR) before comparing. QUIC already saves by the
    # header frame id, so it stays as-is.
    aligned_tmp = None
    if mode in ("rtp", "scream"):
        aligned_tmp = realign_by_qr(tgt_folder)
        tgt_folder = aligned_tmp

    # Step 1 — per-frame video quality (SSIM, PSNR)
    print(f"  Computing SSIM/PSNR on frames 1..{STOP_FRAME-1} (frames dir: {tgt_folder}) ...")
    compare_images(
        ref_folder=REF_FOLDER,
        tgt_folder=tgt_folder,
        start_num=1,
        end_num=STOP_FRAME - 1,
        csv_path=quality_csv,
    )

    # Step 2 — load quality and timing data
    df_q = pd.read_csv(quality_csv)
    df_q["frame_id"] = df_q["frame"].str.replace(".png", "", regex=False).astype(int)

    # Step 2b — per-frame VMAF over the received sequence (one libvmaf pass)
    print(f"  Computing VMAF over {len(df_q)} received frames ...")
    vmaf_map = compute_vmaf_sequence(tgt_folder, df_q["frame_id"].tolist())
    df_q["VMAF"] = df_q["frame_id"].map(vmaf_map)

    df_fps = load_fps_rt(mode)

    df = df_q.merge(df_fps, on="frame_id", how="left")

    # Step 3 — perceived quality (QoE) from VMAF + response time.
    # Formula lives in tools/qoe.py (single source of truth, shared with the GUI).
    df["fps"] = df["fps"].fillna(FPS_TARGET)
    df["QoE"] = perceived_qoe(df["VMAF"], df["response_time_ms"])

    # Step 4 — per-frame metrics on a FIXED frame axis (1..STOP_FRAME-1).
    # Every mode shares the same x-axis regardless of how many frames arrived;
    # frames that were never received appear as gaps (NaN), which the plots
    # leave as breaks in the line instead of interpolating across them.
    cols = ["frame_id", "SSIM", "PSNR", "fps", "response_time_ms", "QoE"]
    if "VMAF" in df.columns:
        cols.insert(3, "VMAF")   # keep VMAF next to the other quality metrics
    per_frame = df[cols].sort_values("frame_id")
    full_idx = pd.DataFrame({"frame_id": range(1, STOP_FRAME)})
    out = full_idx.merge(per_frame, on="frame_id", how="left")

    out.to_csv(metrics_csv, index=False)
    received = int(out["SSIM"].notna().sum())
    print(f"  Saved: {metrics_csv}  ({len(out)} frames, {received} received)")
    print(f"  Avg SSIM={out['SSIM'].mean():.4f}  "
          f"Avg RT={out['response_time_ms'].mean():.1f}ms  "
          f"Avg QoE={out['QoE'].mean():.4f}")

    if aligned_tmp:
        shutil.rmtree(aligned_tmp, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quic", "rtp", "roq", "scream"], required=True)
    args = parser.parse_args()
    compute_metrics(args.mode)
