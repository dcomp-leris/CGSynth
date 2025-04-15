'''
# 

'''

import os, time
import subprocess
import multiprocessing


def run_player1():
    """Run the player Python script."""
    print("Player1 is running ....")
    subprocess.run(["sudo","python3", "/home/leris/mygamer/tofino/player_tofino1.py"], check=True)

'''
def run_player2():
    """Run the player Python script."""
    print("Player2 is running ....")
    subprocess.run(["sudo","python3", "/home/leris/mygamer/tofino/player_tofino2.py"], check=True)


def run_player3():
    """Run the player Python script."""
    print("Player3 is running ....")
    subprocess.run(["sudo","python3", "/home/leris/mygamer/tofino/player_tofino3.py"], check=True)

def run_tshark():
    """Run tshark command that requires sudo privileges."""
    cmd = ["tshark", "-i", "enp2s0np0", "-w", "./mypcap/my.pcap"]
    subprocess.run(cmd, check=True)
    print("Tshark is running ....")


def run_kill_ports():
    subprocess.run(["sudo","/home/leris/mygamer/tofino/port_clean1.sh"], check=True)
    print("killed the ports ***")
    time.sleep(1)

def run_delete_frames1():
    subprocess.run(["sudo","rm", "-f", "/home/leris/mygamer/tofino/rcv_forza_f/*.*"], check=True)
    print("Removed RCV Frames1 ***")
    time.sleep(1)

def run_delete_frames2():
    subprocess.run(["sudo","rm", "-f", "/home/leris/mygamer/tofino/rcv_forza_s/*.*"], check=True)
    print("Removed RCV Frames2 ***")
    time.sleep(1)

def run_delete_frames3():
    subprocess.run(["sudo","rm", "-f", "/home/leris/mygamer/tofino/rcv_forza_t/*.*"], check=True)
    print("Removed RCV Frames3 ***")    
    time.sleep(1)


def run_delete_pcap():
    subprocess.run(["sudo","rm", "-f", "/home/leris/mygamer/tofino/mypcap/my.pcap"], check=True)
    print("Removed PCAP Files ***")
''' 

if __name__ == "__main__":
    run_kill_ports()
    run_delete_pcap()
    run_delete_frames1()
    run_delete_frames2()
    run_delete_frames3()

    player1_process = multiprocessing.Process(target=run_player1)
    player2_process = multiprocessing.Process(target=run_player2)
    player3_process = multiprocessing.Process(target=run_player3)
    tshark_process = multiprocessing.Process(target=run_tshark)
    
    print('\n')
    print('+++++++++++++++++++++++++')
    # Create processes
    
    #scream_process = multiprocessing.Process(target=run_scream_command)
    
    # Start processes
    player1_process.start()
    player2_process.start()
    player3_process.start()
    tshark_process.start()
    
    #scream_process.start()

    # Wait for both to complete
    player1_process.join()
    player2_process.join()
    player3_process.join()
    tshark_process.join()
    #scream_process.join()
