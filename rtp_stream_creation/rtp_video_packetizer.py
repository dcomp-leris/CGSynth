from pathlib import Path
from scapy.all import *
import os
import csv
import subprocess
import tempfile
from collections import defaultdict
import random
import itertools
import itertools

DEFAULT_CODEC = "h264"
MIN_PAYLOAD_SIZE = 900
MAX_PAYLOAD_SIZE_LIMIT = 1400
rtp_clock_rate = 90000  # Hz


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
            start_found = False
            for code in start_codes:
                if data[i:i + len(code)] == code:
                    start = i + len(code)
                    start_found = True
                    break
            
            if not start_found:
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

def load_ipi_values(ipi_file):
    """Load IPI values from CSV file."""
    ipis = []
    try:
        with open(ipi_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                try:
                    # Convert to seconds, don't filter out "Other" packets
                    ipi = float(row[0]) / 100000  # Convert to seconds
                    ipis.append(ipi)
                except Exception as e:
                    print(f"Skipping invalid row: {row} - {e}")
    except Exception as e:
        print(f"Error loading IPI file: {e}")
        return []
    
    if not ipis:
        # Fallback to default values if no valid IPIs found
        print("No valid IPI values found. Using default values.")
        ipis = [0.033] * 100  # ~30 fps
    
    print(f"Loaded {len(ipis)} IPI values")
    return ipis

def packetize_nalu(nalu, seq, current_timestamp, ssrc, marker, max_payload_size, ipi_cycler):
    """
    RTP packetization of a single NAL unit.
    Returns a list of (packet, new_seq, updated_timestamp) tuples.
    
    FIXED: All fragments of the same NAL unit get the same timestamp,
    only updating timestamps between different NAL units.
    """
    packets = []
    # Get next IPI value only once per NAL unit
    ipi = next(ipi_cycler)
    timestamp = current_timestamp + int(ipi * rtp_clock_rate)
    
    if len(nalu) <= max_payload_size:
        # Single NAL Unit packet
        rtp = RTP(
            version=2, padding=0, extension=0, marker=marker,
            payload_type=96, sequence=seq % 65536,
            timestamp=int(timestamp), sourcesync=ssrc
        ) / Raw(load=nalu)
        packets.append((rtp, seq + 1, timestamp))
    else:
        # Fragmentation Units
        nal_header = nalu[0]
        nal_type = nal_header & 0x1F
        nal_nri = (nal_header >> 5) & 0x03

        offset = 1
        while offset < len(nalu):
            end = min(offset + max_payload_size - 2, len(nalu))
            start_bit = 1 if offset == 1 else 0
            end_bit = 1 if end == len(nalu) else 0
            
            # Only mark the last fragment with the marker bit if the original NAL had it
            current_marker = marker if end_bit else 0

            fu_indicator = (nal_nri << 5) | 28  # FU-A
            fu_header = (start_bit << 7) | (end_bit << 6) | nal_type
            fragment = bytes([fu_indicator, fu_header]) + nalu[offset:end]

            rtp = RTP(
                version=2, padding=0, extension=0, marker=current_marker,
                payload_type=96, sequence=seq % 65536,
                timestamp=int(timestamp), sourcesync=ssrc
            ) / Raw(load=fragment)
            
            packets.append((rtp, seq + 1, timestamp))
            seq += 1
            offset = end
        
        print(f"Fragmented NAL unit into {len(packets)} packets with timestamp: {timestamp}")

    return packets, timestamp


def encode_image_to_nalus(img_path, codec=DEFAULT_CODEC):
    """
    Encodes an image to H.264/H.265 and extracts NAL units.
    FIXED: Improved NAL unit extraction with proper handling of start codes.
    """
    assert codec in ["h264", "h265"], "Codec must be 'h264' or 'h265'"
    suffix = "h264" if codec == "h264" else "hevc"

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as out_file:
        encoded_path = out_file.name

    try:
        # Use more conservative encoding parameters
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(img_path),
            "-pix_fmt", "yuv420p",
            "-c:v", codec,
            "-profile:v", "baseline",  # Use baseline profile for better compatibility
            "-preset", "slow",         # Better quality, more compatible
            "-x264opts", "keyint=1:min-keyint=1", # Force every frame to be a keyframe
            "-b:v", "1500k",           # Slightly higher bitrate
            "-maxrate", "2500k",
            "-bufsize", "4000k",
            "-f", "rawvideo", encoded_path
        ], check=True)

        with open(encoded_path, "rb") as f:
            data = f.read()

        # Improved NAL unit extraction
        nalus = []
        i = 0
        start_codes = [b'\x00\x00\x00\x01', b'\x00\x00\x01']

        while i < len(data):
            # Find start code
            start_code_pos = -1
            start_code_len = 0
            
            for code in start_codes:
                pos = data.find(code, i)
                if pos != -1 and (start_code_pos == -1 or pos < start_code_pos):
                    start_code_pos = pos
                    start_code_len = len(code)
            
            if start_code_pos == -1:
                break
                
            # Find the start of the next NAL unit
            next_start_pos = len(data)
            for code in start_codes:
                pos = data.find(code, start_code_pos + start_code_len)
                if pos != -1 and pos < next_start_pos:
                    next_start_pos = pos
            
            # Extract the NAL unit without the start code
            nal_start = start_code_pos + start_code_len
            nal_end = next_start_pos
            nal_unit = data[nal_start:nal_end]
            
            if nal_unit:  # Only add if not empty
                nalus.append(nal_unit)
            
            i = start_code_pos + start_code_len
            if i >= next_start_pos:
                i = next_start_pos
        
        # Log the NAL units we found
        for idx, nalu in enumerate(nalus):
            if nalu:
                nal_type = (nalu[0] & 0x1F) if codec == "h264" else ((nalu[0] >> 1) & 0x3F)
                print(f"NAL {idx}: type={nal_type}, size={len(nalu)} bytes")

        return nalus
    finally:
        os.remove(encoded_path)


def create_rtp_packets(codec=DEFAULT_CODEC):
    """
    Reads PNG frames, encodes them to NALUs, and writes RTP packets to PCAP.
    FIXED: Improved timing and packet creation.
    """
    base_path = Path(__file__).parent.resolve()
    img_dir = base_path.parent / "frame_gen" / "rescaling" / "downscaled_original_frames_from_1920_1080_to_1280_720"
    ipi_file = base_path.parent / "rtp_stream_creation" / "all_packets.txt"
    output_pcap = f"rtp_stream_{codec}.pcap"

    # Network configuration
    server_ip = "192.168.0.10"
    user_ip = "192.168.0.20"
    server_port = 5004
    user_port = 5004
    server_mac = "00:11:22:33:44:55"
    user_mac = "66:77:88:99:aa:bb"

    # RTP setup
    ssrc = 12345
    seq = 0
    timestamp = 0
    all_packets = []

    # Read commands
    commands = read_commands()

    # Validate paths
    if not img_dir.exists():
        print(f"Image directory {img_dir} does not exist.")
        return
    if not ipi_file.exists():
        print(f"IPI file {ipi_file} does not exist.")
        return

    # Load IPI values
    ipis = load_ipi_values(ipi_file)
    if not ipis:
        return
    
    # Create a cyclic iterator for IPIs
    ipi_cycler = itertools.cycle(ipis)

    # Find all PNG files
    image_files = sorted(img_dir.glob("*.png"))
    print(f"Found {len(image_files)} PNG files")

    # Use a more consistent payload size - this helps avoid fragmentation issues
    max_payload_size = 1200  # Consistent size that's still below typical MTU

    # Loop through each frame
    for frame_index, img_path in enumerate(image_files, start=1):
        print(f"Processing frame {frame_index}/{len(image_files)}: {img_path.name}, MaxPayload={max_payload_size}")

        # Encode image to NAL units
        nal_units = encode_image_to_nalus(img_path, codec=codec)
        
        # Process each NAL unit for this frame
        for idx, nalu in enumerate(nal_units):
            # Mark only the last NAL unit of the frame
            marker = 1 if idx == len(nal_units) - 1 else 0
            
            # Packetize the NAL unit
            rtp_packets, updated_timestamp = packetize_nalu(
                nalu, seq, timestamp, ssrc, marker, max_payload_size, ipi_cycler
            )
            timestamp = updated_timestamp  # Update timestamp for next NAL unit
            
            # Create network packets
            for rtp, new_seq, _ in rtp_packets:
                ip = IP(src=server_ip, dst=user_ip)
                udp = UDP(sport=server_port, dport=user_port)
                eth = Ether(src=server_mac, dst=user_mac)
                full_packet = eth / ip / udp / rtp
                all_packets.append(full_packet)
                seq = new_seq

        # Process commands for this frame if any
        if frame_index in commands:
            for command in commands[frame_index]:
                # Get next IPI and update timestamp for command packet
                ipi = next(ipi_cycler)
                timestamp += int(ipi * rtp_clock_rate)
                
                # Use a distinct payload type for command RTP packets
                rtp_command = RTP(
                    version=2, padding=0, extension=0, marker=1,
                    payload_type=97, sequence=seq % 65536,
                    timestamp=int(timestamp), sourcesync=ssrc
                ) / Raw(load=command.encode())

                ip = IP(src=user_ip, dst=server_ip)
                udp = UDP(sport=user_port, dport=server_port)
                eth = Ether(src=user_mac, dst=server_mac)
                full_packet = eth / ip / udp / rtp_command
                all_packets.append(full_packet)
                seq += 1
                print(f"Added command for frame {frame_index}: {command} with timestamp {timestamp}")

    # Write packets to PCAP file
    print(f"Writing {len(all_packets)} packets to {output_pcap}")
    wrpcap(output_pcap, all_packets, nano=True)
    print(f"RTP packet stream generation complete.")

if __name__ == "__main__":
    create_rtp_packets(codec=DEFAULT_CODEC)