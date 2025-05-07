#!/usr/bin/env python3
from scapy.all import *
import os
import subprocess
import tempfile
import argparse
from collections import defaultdict


class RTPVideoExtractor:
    def __init__(self, input_pcap, output_file=None, codec="h264"):
        """
        Initialize the RTP video extractor.
        
        Args:
            input_pcap (str): Path to the input PCAP file
            output_file (str): Path to the output video file (default: output.mp4)
            codec (str): Video codec ('h264' or 'h265')
        """
        self.input_pcap = input_pcap
        self.codec = codec.lower()
        
        if output_file is None:
            output_file = f"output.{self.get_extension()}"
        self.output_file = output_file
        
        # Temporary file to store raw NAL units
        self.temp_raw_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=f".{'h264' if self.codec == 'h264' else 'hevc'}"
        ).name
        
    def get_extension(self):
        """Get the appropriate file extension based on codec."""
        return "mp4"  # MP4 container works for both H.264 and H.265
    
    def process_pcap(self):
        """Extract RTP packets from PCAP and rebuild the video stream."""
        print(f"Reading packets from {self.input_pcap}...")
        
        # Read all packets from the PCAP file
        try:
            packets = rdpcap(self.input_pcap)
        except Exception as e:
            print(f"Error reading PCAP file: {e}")
            return False
        
        print(f"Found {len(packets)} packets in the PCAP file")
        
        # Store packets by sequence number to handle out-of-order delivery
        rtp_packets = {}
        # Track fragmented NAL units (FU-A/FU-B)
        fragments = defaultdict(list)
        # Store complete NAL units in order
        nal_units = []
        
        # Extract RTP packets
        for packet in packets:
            if UDP in packet and Raw in packet:
                udp_payload = bytes(packet[Raw])
                
                # Check if this looks like an RTP packet (version 2)
                if len(udp_payload) >= 12 and (udp_payload[0] >> 6) == 2:
                    # Parse RTP header
                    version = (udp_payload[0] >> 6) & 0x3
                    padding = (udp_payload[0] >> 5) & 0x1
                    extension = (udp_payload[0] >> 4) & 0x1
                    cc = udp_payload[0] & 0x0F
                    marker = (udp_payload[1] >> 7) & 0x1
                    payload_type = udp_payload[1] & 0x7F
                    sequence_number = (udp_payload[2] << 8) | udp_payload[3]
                    timestamp = (udp_payload[4] << 24) | (udp_payload[5] << 16) | \
                                (udp_payload[6] << 8) | udp_payload[7]
                    ssrc = (udp_payload[8] << 24) | (udp_payload[9] << 16) | \
                           (udp_payload[10] << 8) | udp_payload[11]
                    
                    # Calculate header size
                    header_size = 12 + 4 * cc
                    if extension:
                        if len(udp_payload) >= header_size + 4:
                            ext_length = (udp_payload[header_size+2] << 8) | udp_payload[header_size+3]
                            header_size += 4 + 4 * ext_length
                    
                    # Extract RTP payload
                    if len(udp_payload) > header_size:
                        payload = udp_payload[header_size:]
                        rtp_packets[sequence_number] = {
                            'payload': payload,
                            'marker': marker,
                            'timestamp': timestamp
                        }
        
        if not rtp_packets:
            print("No RTP packets found in the PCAP file")
            return False
        
        print(f"Found {len(rtp_packets)} RTP packets")
        
        # Process packets in sequence order
        current_timestamp = None
        
        for seq in sorted(rtp_packets.keys()):
            pkt = rtp_packets[seq]
            payload = pkt['payload']
            marker = pkt['marker']
            timestamp = pkt['timestamp']
            
            # Detect timestamp changes (new frame)
            if current_timestamp is not None and timestamp != current_timestamp:
                # Process any pending fragments before moving to new frame
                for ts in list(fragments.keys()):
                    if ts != timestamp:
                        # Concatenate fragments if we have any
                        if fragments[ts]:
                            reconstructed_nal = self.reconstruct_nal_from_fragments(fragments[ts])
                            if reconstructed_nal:
                                nal_units.append(reconstructed_nal)
                        del fragments[ts]
            
            current_timestamp = timestamp
            
            if len(payload) < 1:
                continue
                
            # For H.264
            if self.codec == "h264":
                nal_type = payload[0] & 0x1F
                
                # Single NAL unit
                if nal_type <= 23:
                    # Add start code and the NAL unit
                    nal_units.append(b'\x00\x00\x00\x01' + payload)
                
                # FU-A (Fragmentation Units)
                elif nal_type == 28 and len(payload) >= 2:
                    fu_header = payload[1]
                    start_bit = (fu_header >> 7) & 0x1
                    end_bit = (fu_header >> 6) & 0x1
                    nal_type = fu_header & 0x1F
                    
                    if start_bit:
                        # Start of fragmented NAL unit
                        fragments[timestamp] = [bytes([payload[0] & 0xE0 | nal_type]) + payload[2:]]
                    elif fragments[timestamp]:
                        # Middle or end fragment
                        fragments[timestamp].append(payload[2:])
                        
                        if end_bit:
                            # End of fragmentation, reconstruct the NAL
                            reconstructed_nal = self.reconstruct_nal_from_fragments(fragments[timestamp])
                            if reconstructed_nal:
                                nal_units.append(reconstructed_nal)
                            del fragments[timestamp]
                
                # STAP-A (Single-time aggregation packet)
                elif nal_type == 24:
                    offset = 1
                    while offset + 2 <= len(payload):
                        size = (payload[offset] << 8) | payload[offset+1]
                        offset += 2
                        if offset + size <= len(payload):
                            nal_units.append(b'\x00\x00\x00\x01' + payload[offset:offset+size])
                        offset += size
            
            # For H.265
            elif self.codec == "h265" and len(payload) >= 2:
                nal_type = (payload[0] >> 1) & 0x3F
                
                # Single NAL unit
                if nal_type < 48:
                    nal_units.append(b'\x00\x00\x00\x01' + payload)
                
                # Fragmentation Units
                elif nal_type == 49 and len(payload) >= 3:
                    fu_header = payload[2]
                    start_bit = (fu_header >> 7) & 0x1
                    end_bit = (fu_header >> 6) & 0x1
                    nal_type = fu_header & 0x3F
                    
                    if start_bit:
                        # Create new NAL header
                        original_header = ((payload[0] & 0x81) | (nal_type << 1)).to_bytes(1, byteorder='big')
                        fragments[timestamp] = [original_header + payload[1:2] + payload[3:]]
                    elif fragments[timestamp]:
                        fragments[timestamp].append(payload[3:])
                        
                        if end_bit:
                            reconstructed_nal = self.reconstruct_nal_from_fragments(fragments[timestamp])
                            if reconstructed_nal:
                                nal_units.append(reconstructed_nal)
                            del fragments[timestamp]
        
        # Process any remaining fragments
        for ts in fragments:
            if fragments[ts]:
                reconstructed_nal = self.reconstruct_nal_from_fragments(fragments[ts])
                if reconstructed_nal:
                    nal_units.append(reconstructed_nal)
        
        # Write NAL units to temporary file
        with open(self.temp_raw_file, 'wb') as f:
            for nal in nal_units:
                f.write(nal)
        
        print(f"Extracted {len(nal_units)} NAL units to {self.temp_raw_file}")
        
        # Convert raw NAL units to a playable video file
        return self.convert_to_video()
    
    def reconstruct_nal_from_fragments(self, fragments):
        """
        Reconstruct a complete NAL unit from fragments.
        
        Args:
            fragments (list): List of NAL unit fragments
            
        Returns:
            bytes: Reconstructed NAL unit with start code
        """
        if not fragments:
            return None
        
        # Concatenate all fragments
        reconstructed = b'\x00\x00\x00\x01'
        for fragment in fragments:
            reconstructed += fragment
        
        return reconstructed
    
    def convert_to_video(self):
        """
        Convert the raw NAL units to a playable video file using FFmpeg.
        
        Returns:
            bool: True if conversion was successful, False otherwise
        """
        print(f"Converting raw {self.codec} data to {self.output_file}...")
        
        try:
            cmd = [
                "ffmpeg", "-y",
                "-loglevel", "error",
                "-i", self.temp_raw_file,
                "-c:v", "copy",
                self.output_file
            ]
            
            process = subprocess.run(cmd, check=True)
            print(f"Successfully converted video to {self.output_file}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error converting video: {e}")
            return False
        finally:
            # Clean up the temporary file
            if os.path.exists(self.temp_raw_file):
                os.remove(self.temp_raw_file)


def main():
    parser = argparse.ArgumentParser(description="Extract video from RTP packets in a PCAP file")
    parser.add_argument("input_pcap", default = "rtp_stream_h264.pcap", help="Input PCAP file ,default: rtp_stream_h264.pcap")
    parser.add_argument("-o", "--output", default = "output.mp4", help="Output video file, default: output.mp4")
    parser.add_argument("-c", "--codec", choices=["h264", "h265"], default="h264",
                        help="Video codec (default: h264)")
    
    args = parser.parse_args()
    
    extractor = RTPVideoExtractor(
        input_pcap=args.input_pcap,
        output_file=args.output,
        codec=args.codec
    )
    
    success = extractor.process_pcap()
    if success:
        print("Video extraction completed successfully!")
    else:
        print("Video extraction failed.")


if __name__ == "__main__":
    main()
    