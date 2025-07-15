import os, time
import subprocess
import multiprocessing


def run_player1():
    """Run the player Python script."""
    print("Player1 is running ....")
    subprocess.run(["python", "cg_gamer1.py"], check=True)
    ## /home/alireza/CGReplay/CGReplay/player/cg_gamer_1.py

def run_tshark():
    """Run tshark command that requires sudo privileges."""
    pcap_tmp = "/tmp/mypcap.pcap"
    pcap_dest = os.path.join(os.getcwd(), "mypcap.pcap")
    cmd = ["tshark", "-i", "cgplayer-eth0", "-w", pcap_tmp]
    try:
        subprocess.run(cmd, check=True)
        print("Tshark is running ....")
    except KeyboardInterrupt:
        print("Tshark interrupted by user.")
    finally:
        # Always try to copy the pcap file to the current directory
        try:
            subprocess.run(["cp", pcap_tmp, pcap_dest], check=True)
            print(f"Copied pcap file to {pcap_dest}")
            # Ensure user has read/write permissions
            subprocess.run(["chmod", "777", pcap_dest], check=True)
            print(f"Set permissions 777 on {pcap_dest}")
        except Exception as e:
            print(f"Failed to copy or set permissions on pcap file: {e}")

if __name__ == "__main__":
    #run_kill_ports()
    #run_delete_pcap()
    #run_delete_frames1()
    #run_delete_frames2()
    #run_delete_frames3()

    player1_process = multiprocessing.Process(target=run_player1)
    tshark_process = multiprocessing.Process(target=run_tshark)
    
    print('\n')
    print('+++++++++++++++++++++++++')
    # Create processes
    
    # Start processes
    player1_process.start()
    tshark_process.start()

    # Wait for both to complete
    player1_process.join()
    tshark_process.join()