from pathlib import Path
from scapy.all import *
import os
import csv
import subprocess
import tempfile
from collections import defaultdict
import random

DEFAULT_CODEC = "h264"
MIN_PAYLOAD_SIZE = 900
MAX_PAYLOAD_SIZE_LIMIT = 1400

timestamp = 0
ipi = 0
rtp_clock_rate = 90000


def read_commands():
    base_path = Path(__file__).parent
    sync_file = base_path.parent / "player" / "syncs" / "sync_kombat.txt"
    commands = defaultdict(list)

    if not sync_file.exists():
        print(f"Warning: {sync_file} does not exist. Continuing without commands.")
        return commands

    print("Reading commands from:", sync_file)
    with sync_file.open('r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("ID"):
                continue
            parts = line.split(",")
            try:
                frame_id = int(parts[0])
                encrypted_command = parts[-1]
                commands[frame_id].append(encrypted_command)
                print(f"Frame ID: {frame_id}, Encrypted Command: {encrypted_command}")
            except ValueError:
                print("Invalid line format:", line)

    return commands


def encode_image_to_nalus(img_path, codec=DEFAULT_CODEC):
    assert codec in ["h264", "h265"], "Codec must be 'h264' or 'h265'"
    suffix = "h264" if codec == "h264" else "hevc"

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as out_file:
        encoded_path = out_file.name

    try:
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(img_path),
            "-pix_fmt", "yuv420p",
            "-c:v", codec,
            "-profile:v", "main",
            "-preset", "medium",
            "-b:v", "1000k",
            "-maxrate", "2000k",
            "-bufsize", "3000k",
            "-f", "rawvideo", encoded_path
        ], check=True)

        with open(encoded_path, "rb") as f:
            data = f.read()

        nalus, i = [], 0
        start_codes = [b'\x00\x00\x00\x01', b'\x00\x00\x01']

        while i < len(data):
            for code in start_codes:
                if data[i:i + len(code)] == code:
                    start = i + len(code)
                    break
            else:
                i += 1
                continue

            end = len(data)
            for j in range(start, len(data)):
                for code in start_codes:
                    if data[j:j + len(code)] == code:
                        end = j
                        break
                if end != len(data):
                    break

            nalus.append(data[start:end])
            i = end

        for idx, nalu in enumerate(nalus):
            if nalu:
                nal_type = (nalu[0] & 0x1F) if codec == "h264" else ((nalu[0] >> 1) & 0x3F)
                print(f"NAL {idx}: type={nal_type}, size={len(nalu)} bytes")

        return nalus
    finally:
        os.remove(encoded_path)


def packetize_nalu(nalu, seq, timestamp, ssrc, marker, max_payload_size, ipi_iter):
    packets = []

    if len(nalu) <= max_payload_size:
        ipi = next(ipi_iter)
        prev_timestamp = timestamp
        timestamp += ipi * rtp_clock_rate
        print(f"Timestamp updated: {prev_timestamp:.2f} + ({ipi:.6f} × {rtp_clock_rate}) = {timestamp:.2f}")
        rtp = RTP(version=2, padding=0, extension=0, marker=marker,
                  payload_type=96, sequence=seq % 65536,
                  timestamp=int(timestamp), sourcesync=ssrc) / Raw(load=nalu)
        packets.append((rtp, seq + 1))
    else:
        nal_header = nalu[0]
        nal_type = nal_header & 0x1F
        nal_nri = (nal_header >> 5) & 0x03
        offset = 1

        while offset < len(nalu):
            end = min(offset + max_payload_size - 2, len(nalu))
            start_bit = 1 if offset == 1 else 0
            end_bit = 1 if end == len(nalu) else 0
            current_marker = marker if end_bit else 0

            fu_indicator = (nal_nri << 5) | 28
            fu_header = (start_bit << 7) | (end_bit << 6) | nal_type
            fragment = bytes([fu_indicator, fu_header]) + nalu[offset:end]

            
            ipi = next(ipi_iter)
            prev_timestamp = timestamp
            timestamp += ipi * rtp_clock_rate
            print(f"Timestamp updated: {prev_timestamp:.2f} + ({ipi:.6f} × {rtp_clock_rate}) = {timestamp:.2f}")
            
            rtp = RTP(version=2, padding=0, extension=0, marker=current_marker,
                      payload_type=96, sequence=seq % 65536,
                      timestamp=int(timestamp), sourcesync=ssrc) / Raw(load=fragment)
            packets.append((rtp, seq + 1))
            seq += 1
            offset = end

    return packets, ipi_iter, timestamp


def create_rtp_packets(codec=DEFAULT_CODEC):
    base_path = Path(__file__).resolve().parent
    img_dir = base_path.parent / "frame_gen" / "rescaling" / "downscaled_original_frames_from_1920_1080_to_1280_720"
    ipi_file = base_path.parent / "rtp_stream_creation" / "all_packets.txt"
    output_pcap = f"rtp_stream_{codec}.pcap"

    # Network config
    server_ip, user_ip = "192.168.0.10", "192.168.0.20"
    server_port, user_port = 5004, 5004
    server_mac, user_mac = "00:11:22:33:44:55", "66:77:88:99:aa:bb"

    # RTP setup
    ssrc = 12345
    seq = 0
    timestamp = 0

    # Read commands
    commands = read_commands()

    # Validate paths
    if not img_dir.exists():
        print(f"Image directory {img_dir} does not exist.")
        return
    if not ipi_file.exists():
        print(f"IPI file {ipi_file} does not exist.")
        return

    # Read IPI values
    ipis = []
    with ipi_file.open("r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            try:
                if str(row[3]) == "Other":
                    continue
                else:
                    ipis.append(float(row[0]) / 100000)
                
            except Exception as e:
                print("Skipping invalid row:", row, "-", e)

    ipi_iter = iter(ipis)
    
    image_files = sorted(img_dir.glob("*.png"))
    print(f"Found {len(image_files)} PNG files.")

    all_packets = []

    for frame_index, img_path in enumerate(image_files, start=1):
        max_payload_size = random.randint(MIN_PAYLOAD_SIZE, MAX_PAYLOAD_SIZE_LIMIT)
        print(f"Encoding frame {frame_index}: {img_path.name}, MaxPayload={max_payload_size}")

        # Encode image into NAL units
        nal_units = encode_image_to_nalus(img_path, codec=codec)

        for idx, nalu in enumerate(nal_units):
            marker = 1 if idx == len(nal_units) - 1 else 0
            rtp_packets, ipi_iter, timestamp = packetize_nalu(nalu, seq, int(timestamp), ssrc, marker, max_payload_size, ipi_iter)

            for rtp, new_seq in rtp_packets:

                eth = Ether(src=server_mac, dst=user_mac)
                ip = IP(src=server_ip, dst=user_ip)
                udp = UDP(sport=server_port, dport=user_port)
                all_packets.append(eth / ip / udp / rtp)

                seq = new_seq

        # Send RTP command packets (backward direction)
        if frame_index in commands:
            for command in commands[frame_index]:
                ipi_iter = iter(ipis)
                ipi = next(ipi_iter)
                old_timestamp = timestamp
                timestamp += ipi * rtp_clock_rate
                print(f"Timestamp updated for command: {old_timestamp:.2f} + ({ipi:.6f} × {rtp_clock_rate}) = {timestamp:.2f}")
                rtp_command = RTP(version=2, padding=0, extension=0, marker=1,
                                  payload_type=97, sequence=seq % 65536,
                                  timestamp=int(timestamp), sourcesync=ssrc) / Raw(load=command.encode())
                eth = Ether(src=user_mac, dst=server_mac)
                ip = IP(src=user_ip, dst=server_ip)
                udp = UDP(sport=user_port, dport=server_port)
                all_packets.append(eth / ip / udp / rtp_command)
                print(f"Executing command for frame {frame_index}: {command}")
                seq += 1

    # You can optionally save the packets to a PCAP file here
    # wrpcap(output_pcap, all_packets)
    print(f"RTP packet stream generation complete. Total packets: {len(all_packets)}")
    print(f"Writing {len(all_packets)} packets to {output_pcap}")
    wrpcap(output_pcap, all_packets, nano=True)


if __name__ == "__main__":
    create_rtp_packets(codec=DEFAULT_CODEC)