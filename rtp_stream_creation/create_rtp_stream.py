from pathlib import Path
from scapy.all import *
import os

MAX_PAYLOAD_SIZE = 1400  # Maximum payload size for RTP packets

def read_commands():
    # Get the current file's directory and build the path safely
    base_path = Path(__file__).parent
    target_file = base_path.parent / "player" / "syncs" / "sync_kombat.txt"
    print("Target file path:", target_file)
    
    # Initialize an empty dictionary to store the sync data
    commands = {}
    
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
                commands[id_] = encrypted_command
            
                print(f"ID: {id_}, Encrypted Command: {encrypted_command}")
    return commands


def create_rtp_packets():
    base_path = Path(__file__).parent
    image_folder = base_path.parent / "frame_gen" / "interpolation_methods" / "original_frames_1920_1080"
    
    output_pcap = "rtp_stream.pcap"
    
    # RTP/UDP/IP/Ethernet parameters
    src_ip = "192.168.0.10"
    dst_ip = "192.168.0.20"
    src_port = 5004
    dst_port = 5004
    ssrc = 12345
    sequence_number = 0  # Initial sequence number
    timestamp = 1000

    packets = []

    # Iterate through sorted image files
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".png"):
            img_path = os.path.join(image_folder, filename)
            with open(img_path, 'rb') as f:
                img_data = f.read()

            # Fragment the image into chunks if it's too large
            for i in range(0, len(img_data), MAX_PAYLOAD_SIZE):
                chunk = img_data[i:i + MAX_PAYLOAD_SIZE]
                
                # Set the marker for the last chunk of the frame
                marker = 1 if i + MAX_PAYLOAD_SIZE >= len(img_data) else 0

                rtp = RTP(
                    version=2,
                    padding=0,
                    extension=0,
                    marker=marker,
                    payload_type=26,  # Or dynamic payload type 96
                    sequence=sequence_number % 65536,  # Ensure sequence number is within range
                    timestamp=timestamp,
                    sourcesync=ssrc
                ) / Raw(load=chunk)

                udp = UDP(sport=src_port, dport=dst_port) / rtp
                ip = IP(src=src_ip, dst=dst_ip) / udp
                eth = Ether(src="00:11:22:33:44:55", dst="66:77:88:99:aa:bb") / ip

                packets.append(eth)

                # Increment sequence number for each chunk
                sequence_number += 1

            # Increment timestamp for the next frame (keep it constant across fragments of the same frame)
            timestamp += 3000  # Assuming ~30fps, adjust accordingly

    # Write to pcap file
    wrpcap(output_pcap, packets)
    print(f"PCAP file created: {output_pcap}")

commands = read_commands()
create_rtp_packets()


"""from scapy.all import *
import os

# Load your still image
with open("your_image.jpg", "rb") as f:
    image_data = f.read()

# Split into RTP-sized chunks
chunk_size = 1400
chunks = [image_data[i:i+chunk_size] for i in range(0, len(image_data), chunk_size)]

# Base values
base_timestamp_udp = 0xabcdef01
base_timestamp_rtp = 123456789
base_seq_num_rtp = 1000

packets = []
num_loops = 5  # send the image 5 times

for loop in range(num_loops):
    for i, chunk in enumerate(chunks):
        seq_num = base_seq_num_rtp + loop * len(chunks) + i
        rtp_timestamp = base_timestamp_rtp + (loop * 90000) + i * 3600  # simulate time per image/frame

        # RTP header
        rtp_header = b'\x80\x60'                            # V=2, PT=96
        rtp_header += seq_num.to_bytes(2, 'big')
        rtp_header += rtp_timestamp.to_bytes(4, 'big')
        rtp_header += (0x12345678).to_bytes(4, 'big')       # SSRC

        payload = rtp_header + chunk

        udp_len = len(payload) + 8  # 8 bytes for UDP header
        ip_len = udp_len + 20       # 20 bytes for IP header

        # Fake packet
        pkt = Ether(
            src="11:22:33:44:55:66",
            dst="aa:bb:cc:dd:ee:ff"
        ) / IP(
            src="192.168.1.10",
            dst="192.168.1.20",
            len=ip_len
        ) / UDP(
            sport=5004,
            dport=5004,
            len=udp_len
        ) / Raw(load=payload)

        packets.append(pkt)

# Save all packets at once
wrpcap("still_image_rtp_stream.pcap", packets)

print(f"Done! {len(packets)} total packets written to still_image_rtp_stream.pcap")
    """