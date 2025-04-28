from pathlib import Path
from scapy.all import *
import os
import csv

MAX_PAYLOAD_SIZE = 1400  # Maximum payload size for RTP packets

def read_commands():
    # Get the current file's directory and build the path safely
    base_path = Path(__file__).parent
    target_file = base_path.parent / "player" / "syncs" / "sync_kombat.txt"
    print("Target file path:", target_file)
    
    # Create a dictionary to hold commands
    # The key will be the command ID, and the value will be a list of commands
    commands = defaultdict(list)
    
    # Check if the file exists
    if not target_file.exists():
        raise FileNotFoundError(f"The file {target_file} does not exist.")
    else:
    # Read the content of the file
        with target_file.open('r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("ID"):
                    continue  # skip empty lines and the header
                parts = line.split(",")
                id_ = int(parts[0])
                encrypted_command = parts[-1]
                commands[id_].append(encrypted_command)
            
                print(f"ID: {id_}, Encrypted Command: {encrypted_command}")

    return commands


def create_rtp_packets():
    base_path = Path(__file__).parent
    image_folder = base_path.parent / "frame_gen" / "original_frames" / "mortal_kombat_11" / "1920_1080"
    
    output_pcap = "rtp_stream.pcap"

    # Server → User
    server_ip = "192.168.0.10"
    user_ip = "192.168.0.20"
    server_port = 5004
    user_port = 5004
    server_mac = "00:11:22:33:44:55"
    user_mac = "66:77:88:99:aa:bb"

    ssrc = 12345
    sequence_number = 0
    timestamp = 1000

    packets = []

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
    print(f"Found {len(image_files)} image files in {image_folder}")

    images_sent = 0

    for filename in image_files:
        img_path = os.path.join(image_folder, filename)
        with open(img_path, 'rb') as f:
            img_data = f.read()

        # Fragment image into RTP chunks
        for i in range(0, len(img_data), MAX_PAYLOAD_SIZE):
            chunk = img_data[i:i + MAX_PAYLOAD_SIZE]
            marker = 1 if i + MAX_PAYLOAD_SIZE >= len(img_data) else 0

            rtp = RTP(
                version=2,
                padding=0,
                extension=0,
                marker=marker,
                payload_type=26,
                sequence=sequence_number % 65536,
                timestamp=timestamp,
                sourcesync=ssrc
            ) / Raw(load=chunk)

            udp = UDP(sport=server_port, dport=user_port) / rtp
            ip = IP(src=server_ip, dst=user_ip) / udp
            eth = Ether(src=server_mac, dst=user_mac) / ip

            packets.append(eth)
            sequence_number += 1

        images_sent += 1
        timestamp += 3000  # advance per frame

        # Check if commands exist for this frame ID
        if images_sent in commands:
            for cmd in commands[images_sent]:
                print(f"Triggering response for ID {images_sent}: {cmd}")

                # Simulated RTP response from User → Server
                response_rtp = RTP(
                    version=2,
                    padding=0,
                    extension=0,
                    marker=1,
                    payload_type=97,
                    sequence=sequence_number % 65536,
                    timestamp=timestamp,
                    sourcesync=ssrc + 1
                ) / Raw(load=cmd.encode())

                response_udp = UDP(sport=user_port, dport=server_port) / response_rtp
                response_ip = IP(src=user_ip, dst=server_ip) / response_udp
                response_eth = Ether(src=user_mac, dst=server_mac) / response_ip

                packets.append(response_eth)
                sequence_number += 1

    # Write all packets to PCAP
    wrpcap(output_pcap, packets)
    print(f"PCAP file created with {len(packets)} packets: {output_pcap}")


commands = read_commands()
create_rtp_packets()