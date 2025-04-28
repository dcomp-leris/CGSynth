from pathlib import Path
from scapy.all import rdpcap
import os
import struct
import binascii

def extract_png_packets(pcap_file="rtp_stream.pcap", output_dir="extracted_png"):
    """
    Extract only PNG images (payload type 26) from a PCAP file containing RTP packets
    
    Args:
        pcap_file: Path to PCAP file
        output_dir: Directory to save extracted PNG images
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create a diagnostics directory
    diag_dir = os.path.join(output_dir, "diagnostics")
    Path(diag_dir).mkdir(exist_ok=True)
    
    # Open diagnostics log file
    diag_log = open(os.path.join(diag_dir, "extraction_log.txt"), "w")
    diag_log.write("PNG Extraction Log\n")
    diag_log.write("================\n\n")
    
    # Read the PCAP file
    print(f"Reading PCAP file: {pcap_file}")
    packets = rdpcap(pcap_file)
    print(f"Found {len(packets)} packets in the PCAP file")
    
    # Keep track of current frame being built
    current_frames = {}  # SSRC -> (timestamp, frame_data)
    completed_frames = []  # List of (timestamp, ssrc, frame_data)
    
    # Process each packet
    for packet in packets:
        # Check if packet has UDP layer
        if packet.haslayer('UDP') and packet.haslayer('Raw'):
            # Extract RTP header from raw payload
            raw_data = bytes(packet['Raw'])
            if len(raw_data) < 12:  # RTP header is at least 12 bytes
                continue
            
            try:
                # First byte contains version, padding, extension, and CSRC count
                first_byte = raw_data[0]
                version = (first_byte >> 6) & 0x3
                padding = (first_byte >> 5) & 0x1
                extension = (first_byte >> 4) & 0x1
                cc = first_byte & 0xF
                
                # Second byte contains marker and payload type
                second_byte = raw_data[1]
                marker = (second_byte >> 7) & 0x1
                payload_type = second_byte & 0x7F
                
                # ONLY process packets with payload type 26 (PNG)
                if payload_type != 26:
                    continue
                
                # Extract sequence number, timestamp, and SSRC
                sequence = struct.unpack('!H', raw_data[2:4])[0]
                timestamp = struct.unpack('!I', raw_data[4:8])[0]
                ssrc = struct.unpack('!I', raw_data[8:12])[0]
                
                # Extract payload (skip RTP header)
                header_size = 12 + (4 * cc)
                
                # Handle extension header if present
                if extension:
                    if len(raw_data) < header_size + 4:
                        continue
                    ext_header_len = struct.unpack('!H', raw_data[header_size+2:header_size+4])[0] * 4
                    header_size += 4 + ext_header_len
                
                # Extract payload
                if len(raw_data) <= header_size:
                    continue
                payload = raw_data[header_size:]
                
                # Initialize frame data if needed
                if ssrc not in current_frames:
                    current_frames[ssrc] = (timestamp, [])
                
                # Check if this packet belongs to a new frame
                current_ts, frame_data = current_frames[ssrc]
                if current_ts != timestamp:
                    # If we were building a frame but never got the marker bit, save it anyway
                    if frame_data:
                        sorted_data = sorted(frame_data, key=lambda x: x[0])
                        completed_data = b''.join([p[1] for p in sorted_data])
                        completed_frames.append((current_ts, ssrc, completed_data))
                        
                        # Log the frame completion
                        diag_log.write(f"Frame completed due to timestamp change: TS={current_ts}, SSRC={ssrc}\n")
                        diag_log.write(f"  Packets: {len(frame_data)}, First seq: {sorted_data[0][0]}, Last seq: {sorted_data[-1][0]}\n\n")
                    
                    # Start a new frame
                    current_frames[ssrc] = (timestamp, [])
                    current_ts, frame_data = current_frames[ssrc]
                
                # Add this packet's payload to the current frame
                current_frames[ssrc][1].append((sequence, payload))
                
                # If this packet has the marker bit set, it's the last packet in the frame
                if marker:
                    # Sort packets by sequence number and extract payload
                    sorted_data = sorted(frame_data, key=lambda x: x[0])
                    completed_data = b''.join([p[1] for p in sorted_data])
                    
                    # Save the completed frame
                    completed_frames.append((timestamp, ssrc, completed_data))
                    
                    # Log the frame completion
                    diag_log.write(f"Frame completed due to marker bit: TS={timestamp}, SSRC={ssrc}\n")
                    diag_log.write(f"  Packets: {len(frame_data)}, First seq: {sorted_data[0][0]}, Last seq: {sorted_data[-1][0]}\n\n")
                    
                    # Clear the current frame
                    current_frames[ssrc] = (timestamp, [])
                
            except Exception as e:
                print(f"Error processing packet: {e}")
                diag_log.write(f"Error processing packet: {e}\n")
                continue
    
    # Handle any remaining incomplete frames
    for ssrc, (timestamp, frame_data) in current_frames.items():
        if frame_data:
            sorted_data = sorted(frame_data, key=lambda x: x[0])
            completed_data = b''.join([p[1] for p in sorted_data])
            completed_frames.append((timestamp, ssrc, completed_data))
            
            # Log the frame completion
            diag_log.write(f"Frame completed at end of processing: TS={timestamp}, SSRC={ssrc}\n")
            diag_log.write(f"  Packets: {len(frame_data)}, First seq: {sorted_data[0][0]}, Last seq: {sorted_data[-1][0]}\n\n")
    
    # Sort frames by timestamp
    completed_frames.sort(key=lambda x: x[0])
    
    # Save the completed frames
    frame_count = 0
    for timestamp, ssrc, image_data in completed_frames:
        # Skip empty frames
        if not image_data:
            continue
        
        # Check if the data actually contains PNG magic bytes
        if len(image_data) >= 8 and image_data[:8] == b'\x89PNG\r\n\x1a\n':
            image_type = "png"
        # Look for PNG chunks elsewhere in case header is corrupted
        elif len(image_data) >= 12:
            # Check for IHDR chunk which should be right after the PNG signature
            png_sig_found = False
            for i in range(min(100, len(image_data) - 12)):
                if image_data[i:i+8] == b'\x89PNG\r\n\x1a\n':
                    # Found PNG signature at an offset - log and fix
                    diag_log.write(f"Frame {frame_count}: Found PNG signature at offset {i}\n")
                    image_data = image_data[i:]  # Fix by keeping only from PNG signature onward
                    png_sig_found = True
                    break
                elif image_data[i:i+4] == b'IHDR':
                    diag_log.write(f"Frame {frame_count}: Found IHDR at offset {i}, may be PNG with corrupted header\n")
                    # Save the problematic header for analysis
                    with open(os.path.join(diag_dir, f"frame_{frame_count:04d}_header.bin"), "wb") as f:
                        f.write(image_data[:min(100, len(image_data))])
                    png_sig_found = True
                    break
            
            if not png_sig_found:
                # This doesn't look like a PNG - save as bin for analysis
                diag_file = os.path.join(diag_dir, f"frame_{frame_count:04d}_{timestamp}.bin")
                with open(diag_file, 'wb') as f:
                    f.write(image_data)
                diag_log.write(f"  Frame {frame_count}: No PNG signature found, saved to {diag_file}\n\n")
                frame_count += 1
                continue
        else:
            # Too small to be a valid PNG
            diag_file = os.path.join(diag_dir, f"frame_{frame_count:04d}_{timestamp}.bin")
            with open(diag_file, 'wb') as f:
                f.write(image_data)
            diag_log.write(f"  Frame {frame_count}: Too small to be PNG ({len(image_data)} bytes), saved to {diag_file}\n\n")
            frame_count += 1
            continue
        
        # Save the PNG image
        output_file = os.path.join(output_dir, f"frame_{frame_count:04d}_{timestamp}.png")
        with open(output_file, 'wb') as f:
            f.write(image_data)
        print(f"Saved {output_file} ({len(image_data)} bytes)")
        
        # Log frame info
        diag_log.write(f"Frame {frame_count}: TS={timestamp}, Size={len(image_data)}, Type=png\n")
        if len(image_data) >= 16:
            diag_log.write(f"  First 16 bytes: {binascii.hexlify(image_data[:16])}\n\n")
        
        frame_count += 1
    
    print(f"Extracted {frame_count} PNG frames to {output_dir}")
    diag_log.write(f"\nExtraction completed: {frame_count} PNG frames processed\n")
    diag_log.close()
    
    print(f"Diagnostic information saved to {diag_dir}")
    return frame_count

if __name__ == "__main__":
    print("PNG RTP Packet Extractor")
    print("------------------------")
    
    pcap_file = input("Enter PCAP file path (default: rtp_stream.pcap): ").strip() or "rtp_stream.pcap"
    output_dir = input("Enter output directory (default: extracted_frames): ").strip() or "extracted_frames"
    
    extract_png_packets(pcap_file, output_dir)