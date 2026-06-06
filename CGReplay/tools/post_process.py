#!/usr/bin/env python3
"""
CGReplay post-processing — video quality + QoE metrics.

Run from CGReplay/ root:
    python3 tools/post_process.py --mode quic
    python3 tools/post_process.py --mode rtp
    python3 tools/post_process.py --mode scream

Outputs: player/logs/metrics_{mode}.csv
Columns: second, SSIM, PSNR, fps, response_time_ms, QoE
"""

import argparse
import os
import sys
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from video_Quality import compare_images

CONFIG_PATH   = "config/config.yaml"
REF_FOLDER    = os.path.join("server", "Kombat")
LOGS_DIR      = os.path.join("player", "logs")

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

STOP_FRAME  = cfg["Running"]["stop_frm_number"]
FPS_TARGET  = cfg["encoding"]["fps"]


def load_fps_rt(mode: str) -> pd.DataFrame:
    """Return DataFrame with frame_id, fps, response_time_ms columns."""
    if mode == "quic":
        path = os.path.join(LOGS_DIR, "ply_quic_frame.csv")
        df = pd.read_csv(path)
        df = df[["frame_id", "fps", "response_time_ms", "recv_time"]]
        df = df.rename(columns={"recv_time": "timestamp"})
    else:
        # RTP and SCReAM use ply_frame.csv + responsetime_CG.csv
        frame_path = os.path.join(LOGS_DIR, "ply_frame.csv")
        rt_path    = os.path.join(LOGS_DIR, "responsetime_CG.csv")
        df_frame   = pd.read_csv(frame_path)[["frame_id", "fps"]]
        df_rt      = pd.read_csv(rt_path)
        df_rt["response_time_ms"] = (
            (df_rt["cmd_timestamp"] - df_rt["frame_timestamp"]) * 1000.0
        )
        df = df_frame.merge(df_rt[["frame_id", "frame_timestamp", "response_time_ms"]],
                            on="frame_id", how="left")
        df = df.rename(columns={"frame_timestamp": "timestamp"})

    df = df.sort_values("frame_id").reset_index(drop=True)
    return df


def compute_metrics(mode: str):
    print(f"[post_process] mode={mode}")

    quality_csv = os.path.join(LOGS_DIR, f"quality_{mode}.csv")
    metrics_csv = os.path.join(LOGS_DIR, f"metrics_{mode}.csv")

    # Use mode-specific backup if it exists, otherwise fall back to shared dir
    tgt_mode = os.path.join("player", f"received_frames_{mode}")
    tgt_fallback = os.path.join("player", "received_frames")
    tgt_folder = tgt_mode if os.path.isdir(tgt_mode) else tgt_fallback

    # Step 1 — per-frame video quality (SSIM, PSNR, VMAF if available)
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
    df_fps = load_fps_rt(mode)

    df = df_q.merge(df_fps, on="frame_id", how="left")

    # Step 3 — QoE = SSIM × (fps / fps_target)  [0..1 scale]
    df["fps"] = df["fps"].fillna(FPS_TARGET)
    df["response_time_ms"] = df["response_time_ms"].fillna(0)
    df["QoE"] = df["SSIM"] * (df["fps"].clip(upper=FPS_TARGET) / FPS_TARGET)

    # Step 4 — bucket into 1-second windows
    # For RTP/SCReAM, responsetime_CG.csv only has command frames — most rows lack a timestamp.
    # Fill missing timestamps using frame_id / fps as a monotonic estimate.
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        t0 = df["timestamp"].dropna().min()
        df["timestamp"] = df["timestamp"].fillna(df["frame_id"] / FPS_TARGET + t0)
        df["second"] = (df["timestamp"] - t0).clip(lower=0).astype(int)
    else:
        df["second"] = (df["frame_id"] / FPS_TARGET).astype(int)

    agg = df.groupby("second").agg(
        SSIM=("SSIM", "mean"),
        PSNR=("PSNR", "mean"),
        fps=("fps", "mean"),
        response_time_ms=("response_time_ms", "mean"),
        QoE=("QoE", "mean"),
    ).reset_index()

    agg.to_csv(metrics_csv, index=False)
    print(f"  Saved: {metrics_csv}  ({len(agg)} seconds)")
    print(f"  Avg SSIM={agg['SSIM'].mean():.4f}  "
          f"Avg RT={agg['response_time_ms'].mean():.1f}ms  "
          f"Avg QoE={agg['QoE'].mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quic", "rtp", "scream"], required=True)
    args = parser.parse_args()
    compute_metrics(args.mode)
