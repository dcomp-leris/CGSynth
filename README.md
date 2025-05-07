# CGReplay


# RTP Video Tools

A pair of Python scripts for working with video streams over RTP networks. These tools enable you to create packet captures (PCAPs) of H.264/H.265 video streams and extract the original video from such captures.

## Overview

- **rtp_video_packetizer.py**: Converts a series of PNG images into an H.264/H.265 video stream, creates RTP packets for network transmission, and stores them in a PCAP file.
- **rtp_video_extractor.py**: Extracts H.264/H.265 video from RTP packets in a PCAP file and reconstructs the original video stream.

## Requirements

- Python 3.6+
- Scapy (`pip install scapy`)
- FFmpeg (must be installed and available in your PATH)

Additional Python dependencies:
```
pip install scapy
```

## Usage

### Creating RTP Video Packets (PCAP)

```bash
python rtp_video_packetizer.py
```

This script will:
1. Read PNG frames from the specified directory
2. Encode them to H.264/H.265 using FFmpeg
3. Packetize the encoded video into RTP packets
4. Save the packets as a PCAP file

Configuration options are defined within the script, including:
- Image source directory
- Output PCAP filename
- Network parameters (IP addresses, ports)
- Video codec (H.264 or H.265)

### Extracting Video from PCAP

```bash
python rtp_video_extractor.py input_pcap.pcap [-o output_video.mp4] [-c codec]
```

Arguments:
- `input_pcap.pcap`: Path to the input PCAP file (default: 'rtp_stream_h264.pcap')
- `-o, --output`: Path to the output video file (default: 'output.mp4')
- `-c, --codec`: Video codec ('h264' or 'h265', default: 'h264')

Example:
```bash
python rtp_video_extractor.py rtp_stream_h264_fixed.pcap -o extracted_video.mp4 -c h264
```

## How It Works

### Packetizer
1. Reads PNG image frames from a directory
2. Encodes each frame to H.264/H.265 using FFmpeg
3. Extracts NAL units from the encoded video
4. Packetizes NAL units into RTP packets according to RFC standards
5. Creates Ethernet, IP, and UDP headers for the packets
6. Writes the complete packets to a PCAP file

### Extractor
1. Reads RTP packets from a PCAP file
2. Extracts video payload from each packet
3. Reconstructs fragmented NAL units
4. Builds a complete H.264/H.265 bitstream with proper start codes
5. Uses FFmpeg to convert the raw bitstream to a playable MP4 file

## Notes

- The packetizer assumes PNG files are named and ordered sequentially
- The extractor handles out-of-order packets and fragmented NAL units
- Both scripts support H.264 and H.265 codecs
- The packetizer code has additional functionality for reading encrypted commands from a sync file (optional)

## Troubleshooting

- Ensure FFmpeg is properly installed and available in your PATH
- Check that your PNG images are valid and can be encoded by FFmpeg
- For extraction problems, verify that the PCAP contains valid RTP packets with H.264/H.265 payload
- Malformed or incomplete packets in the PCAP file may result in corrupted video output