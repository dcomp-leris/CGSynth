#!/usr/bin/env python3
"""
Standalone live frame viewer — runs under the SYSTEM Python (GTK OpenCV).

The QUIC receiver runs under the venv, whose OpenCV uses the Qt5 backend. Qt5
does not render reliably when launched as root inside a Mininet namespace,
while the system's GTK build does — it is the same path RTP/SCReAM use to show
live video. So the QUIC receiver decodes frames (venv) and writes them to
received_frames/, and this separate process displays the newest one (system).

Usage:
    /usr/bin/python3 frame_viewer.py <frames_dir>

Stops when /tmp/quic_viewer_stop appears, or after 15 s with no new frame.
"""

import cv2
import os
import sys
import time
import glob

FRAMES_DIR = sys.argv[1] if len(sys.argv) > 1 else "./received_frames"
STOP_FILE  = "/tmp/quic_viewer_stop"
WINDOW     = "CGReplay QUIC — Live Game Video Stream"
IDLE_LIMIT = 15.0  # seconds without a new frame before giving up

last_shown  = None
last_change = time.time()

while True:
    if os.path.exists(STOP_FILE):
        break

    files = glob.glob(os.path.join(FRAMES_DIR, "*.png"))
    if files:
        newest = max(files, key=os.path.getmtime)
        if newest != last_shown:
            img = cv2.imread(newest)
            if img is not None:
                cv2.imshow(WINDOW, img)
                last_shown = newest
                last_change = time.time()

    # waitKey doubles as the ~30 ms frame pacing and the GUI event pump
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

    if last_shown is not None and time.time() - last_change > IDLE_LIMIT:
        break

cv2.destroyAllWindows()
