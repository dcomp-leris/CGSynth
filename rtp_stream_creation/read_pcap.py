from pathlib import Path
from scapy.all import rdpcap, RTP
import struct
import os

def extract_images_from_pcap(pcap_file="rtp_stream.pcap", output_dir="extracted_frames"):
    """
    Extract PNG images from RTP packets in a PCAP file.
    
    Args:
        pcap_file: Path to the PCAP file containing RTP stream
        output_dir: Directory where extracted frames will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the pcap file
    print(f"Reading PCAP file: {pcap_file}")
    packets = rdpcap(pcap_file)
    print(f"Total packets: {len(packets)}")
    
    # Dictionary to store frames by timestamp
    frames_by_timestamp = {}
    
    # Check if Scapy has built-in RTP layer support in this version
    has_rtp_layer = False
    
    for i, packet in enumerate(packets[:10]):  # Check first 10 packets
        if packet.haslayer('RTP'):
            has_rtp_layer = True
            break
    
    print(f"Using {'built-in' if has_rtp_layer else 'manual'} RTP parsing")
    
    # First pass: collect all fragments by timestamp
    for i, packet in enumerate(packets):
        if has_rtp_layer and packet.haslayer('RTP'):
            # Using Scapy's built-in RTP layer
            rtp = packet['RTP']
            timestamp = rtp.timestamp
            marker = rtp.marker
            payload = bytes(packet['RTP'].payload)
            
        elif packet.haslayer('UDP') and packet.haslayer('Raw'):
            # Manual RTP parsing
            udp_payload = bytes(packet['Raw'])
            
            # Skip if too short to be RTP
            if len(udp_payload) < 12:
                continue
                
            # Parse RTP header
            first_byte = udp_payload[0]
            version = (first_byte >> 6) & 0x03
            
            # Verify this is likely an RTP packet
            if version != 2:
                continue
                
            # Extract RTP header fields
            padding = (first_byte >> 5) & 0x01
            extension = (first_byte >> 4) & 0x01
            cc = first_byte & 0x0F
            
            second_byte = udp_payload[1]
            marker = (second_byte >> 7) & 0x01
            payload_type = second_byte & 0x7F
            
            sequence_number = struct.unpack('!H', udp_payload[2:4])[0]
            timestamp = struct.unpack('!I', udp_payload[4:8])[0]
            ssrc = struct.unpack('!I', udp_payload[8:12])[0]
            
            # Calculate header length (12 bytes + CSRC identifiers if any)
            header_length = 12 + (4 * cc)
            
            # Handle RTP header extensions if present
            if extension:
                if len(udp_payload) < header_length + 4:
                    continue
                ext_header = struct.unpack('!HH', udp_payload[header_length:header_length+4])
                header_length += 4 + (ext_header[1] * 4)
            
            # Get the actual payload (after the header)
            if header_length < len(udp_payload):
                payload = udp_payload[header_length:]
            else:
                continue
        
        else:
            # Not an RTP packet
            continue
        
        # Store this fragment indexed by timestamp
        if timestamp not in frames_by_timestamp:
            frames_by_timestamp[timestamp] = {
                'fragments': [],
                'markers': []
            }
        
        # Add this fragment and whether it has a marker
        frames_by_timestamp[timestamp]['fragments'].append(payload)
        frames_by_timestamp[timestamp]['markers'].append(marker)
    
    # Second pass: reassemble and save frames
    frame_count = 0
    for timestamp, frame_data in sorted(frames_by_timestamp.items()):
        # Concatenate all fragments for this timestamp
        full_data = bytearray()
        for fragment in frame_data['fragments']:
            full_data.extend(fragment)
        
        # Look for PNG signature anywhere in the first 100 bytes
        png_start = -1
        for i in range(min(len(full_data) - 8, 100)):
            if full_data[i:i+8] == b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A':
                png_start = i
                break
        
        if png_start >= 0:
            # Found PNG signature, extract from there
            png_data = full_data[png_start:]
            frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
            with open(frame_path, 'wb') as f:
                f.write(png_data)
            print(f"Saved frame {frame_count} to {frame_path} ({len(png_data)} bytes)")
            frame_count += 1
    
    print(f"Extraction complete. {frame_count} PNG frames extracted to {output_dir}")
    
    return frame_count

if __name__ == "__main__":
    extract_images_from_pcap()