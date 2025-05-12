from pathlib import Path
from scapy.all import *
import os
import csv
import subprocess
import tempfile
from collections import defaultdict

MAX_PAYLOAD_SIZE = 1400  # RTP max payload size


def read_commands():
    """
    Reads encrypted commands from sync_kombat.txt.
    Returns a dictionary mapping frame ID to a list of encrypted commands.
    """
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


def encode_image_to_nalus(img_path, codec="h264"):
    """
    Encodes an image to H.264/H.265 and extracts NAL units.
    """
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

        # Extract NAL units
        nalus = []
        i = 0
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


def packetize_h264_nalu(nalu, seq, timestamp, ssrc, marker=0):
    """
    RTP packetization of a single H.264 NAL unit.
    """
    packets = []

    if len(nalu) <= MAX_PAYLOAD_SIZE:
        rtp = RTP(
            version=2, padding=0, extension=0, marker=marker,
            payload_type=96, sequence=seq % 65536,
            timestamp=timestamp, sourcesync=ssrc
        ) / Raw(load=nalu)
        packets.append((rtp, seq + 1))
    else:
        nal_header = nalu[0]
        nal_type = nal_header & 0x1F
        nal_nri = (nal_header >> 5) & 0x03

        offset = 1
        while offset < len(nalu):
            end = min(offset + MAX_PAYLOAD_SIZE - 2, len(nalu))
            start_bit = 1 if offset == 1 else 0
            end_bit = 1 if end == len(nalu) else 0
            current_marker = marker if end_bit else 0

            fu_indicator = (nal_nri << 5) | 28
            fu_header = (start_bit << 7) | (end_bit << 6) | nal_type
            fragment = bytes([fu_indicator, fu_header]) + nalu[offset:end]

            rtp = RTP(
                version=2, padding=0, extension=0, marker=current_marker,
                payload_type=96, sequence=seq % 65536,
                timestamp=timestamp, sourcesync=ssrc
            ) / Raw(load=fragment)
            packets.append((rtp, seq + 1))
            seq += 1
            offset = end

    return packets


def create_rtp_packets(codec="h264"):
    """
    Reads PNG frames, encodes them to NALUs, and writes RTP packets to PCAP.
    Uses inter-packet intervals from a file to calculate RTP timestamps.
    """
    base_path = Path(__file__).parent
    img_dir = base_path.parent / "frame_gen" / "interpolation" / "downscaled_original_frames_from_1920_1080_to_1280_720"
    ipi_file = base_path.parent / "rtp_stream_creation" / "all_packets.txt"
    output_pcap = f"rtp_stream_{codec}.pcap"

    server_ip = "192.168.0.10"
    user_ip = "192.168.0.20"
    server_port = 5004
    user_port = 5004
    server_mac = "00:11:22:33:44:55"
    user_mac = "66:77:88:99:aa:bb"

    ssrc = 12345
    seq = 0
    timestamp = 0
    all_packets = []

    commands = read_commands()

    if not img_dir.exists():
        print(f"Image directory {img_dir} does not exist.")
        return

    if not ipi_file.exists():
        print(f"IPI file {ipi_file} does not exist.")
        return

    # Read IPIs from file
    ipis = []
    with ipi_file.open("r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            try:
                ipi = float(row[0]) / 100000  # Convert to seconds
                ipis.append(ipi)
            except Exception as e:
                print("Skipping invalid row:", row, "-", e)

    image_files = sorted(img_dir.glob("*.png"))
    print(f"Found {len(image_files)} PNG files.")

    if len(ipis) < len(image_files):
        print(f"Warning: Not enough IPI values ({len(ipis)}) for {len(image_files)} frames.")
        return

    rtp_clock_rate = 90000  # Hz

    for frame_index, (img_path, ipi) in enumerate(zip(image_files, ipis), start=1):
        print(f"Encoding frame {frame_index}: {img_path.name} with IPI={ipi:.6f} seconds")
        nal_units = encode_image_to_nalus(img_path, codec=codec)

        for idx, nalu in enumerate(nal_units):
            marker = 1 if idx == len(nal_units) - 1 else 0
            rtp_packets = packetize_h264_nalu(nalu, seq, int(timestamp), ssrc, marker)
            for rtp, new_seq in rtp_packets:
                ip = IP(src=server_ip, dst=user_ip)
                udp = UDP(sport=server_port, dport=user_port)
                eth = Ether(src=server_mac, dst=user_mac)
                full_packet = eth / ip / udp / rtp
                all_packets.append(full_packet)
                old_seq = seq
                seq = new_seq

        # Increment timestamp based on IPI
        timestamp += ipi * rtp_clock_rate  # Convert seconds to RTP timestamp units
        
        if frame_index in commands:
            for command in commands[frame_index]:
                # Use a distinct payload type for command RTP packets (e.g., 97)
                rtp_command = RTP(
                    version=2, padding=0, extension=0, marker=1,
                    payload_type=97, sequence=old_seq % 65536,
                    timestamp=int(timestamp), sourcesync=ssrc
                ) / Raw(load=command.encode())

                ip = IP(src=user_ip, dst=server_ip)
                udp = UDP(sport=user_port, dport=server_port)
                eth = Ether(src=user_mac, dst=server_mac)
                full_packet = eth / ip / udp / rtp_command
                all_packets.append(full_packet)

                print(f"Executing command for frame {frame_index}: {command}")

    print(f"Writing {len(all_packets)} packets to {output_pcap}")
    wrpcap(output_pcap, all_packets)

if __name__ == "__main__":
    create_rtp_packets(codec="h264")
