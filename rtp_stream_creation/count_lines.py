from scapy.all import rdpcap

def count_lines_in_file(file_path):
    try:
        with open(file_path, 'r') as file:
            line_count = sum(1 for _ in file)
        return line_count
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0


def find_min_max_packet_sizes_over_threshold(pcap_file, threshold=900):
    try:
        packets = rdpcap(pcap_file)
        sizes = [len(pkt) for pkt in packets if len(pkt) >= threshold]

        if not sizes:
            print(f"No packets with length >= {threshold} found.")
            return None, None

        min_len = min(sizes)
        max_len = max(sizes)
        return min_len, max_len

    except FileNotFoundError:
        print(f"PCAP file not found: {pcap_file}")
        return None, None
    except Exception as e:
        print(f"An error occurred while reading PCAP: {e}")
        return None, None

if __name__ == "__main__":
    file_name = "all_packets.txt"
    lines = count_lines_in_file(file_name)
    print(f"The file '{file_name}' contains {lines} lines.")
    pcap_file = "ex38.pcap"
    print(f"Finding packet length bounds for packets ≥ 900 bytes in '{pcap_file}'...")
    min_len, max_len = find_min_max_packet_sizes_over_threshold(pcap_file)
    
    if min_len is not None:
        print(f"Minimum packet length ≥ 900: {min_len}")
        print(f"Maximum packet length ≥ 900: {max_len}")
    
