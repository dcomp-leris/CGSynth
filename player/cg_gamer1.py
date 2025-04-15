'''
# Date: 2024-11-29
# Author: Alireza Shirmarz
# Lab: LERIS/UFScar 
# This is Configured for Netsoft 2025 Conference!
# Gamer: (1) 
'''

import cv2, os, socket, time, yaml, threading, subprocess
import pandas as pd
from datetime import datetime
from pyzbar import pyzbar
from collections import deque


os.sched_setaffinity(0, {0})

# Load configuration from YAML file
with open("../config/config.yaml", "r") as file:
#with open("/home/alireza/CG_Repository/CGReplay/config/config.yaml") as file:
    config = yaml.safe_load(file)

# game name
game_name = config["Running"]["game"]
stop_frm_number = config["Running"]["stop_frm_number"]


# server setup
cg_server_ip = config["server"]["server_IP"]        # CG Server IP address
cg_server_port = config["server"]["server_port"]    # Port for receiving control (Joystick) commands from player

# client (player) setup
player_ip = config['gamer']["player_IP"]                     # CG Gamer IP address
player_port =config['gamer']["player_streaming_port"]       # UDP Port for streaming video to Gamer
my_command_port = config['gamer']["palyer_command_port"]

# sync setup
folder_path = config[game_name]["frames"] 
sync_file = config[game_name]["sync_file"]  

# log setup 
rate_log = config["gamer"]["player_rate_log"] 
time_log = config["gamer"]["player_time_log"]
received_frames = config["gamer"]["received_frames"]

player_interface = config["gamer"]["player_interface"]

# Do you want to watch the Game Video live? 
live_watching = config["Running"]["live_watching"]



# Ack Rate
ack_freq = config["sync"]["ack_freq"]


# Scream enable or disable
scream_state=config["protocols"]["SCReAM"]   
scream_receiver=config["protocols"]["receiver"]

# Custom function to load autocommands.txt while handling the complex 'command' field
def load_syncfile(file_path):
    autocommands = []
    with open(file_path, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            # Split only on the last comma to avoid splitting inside the 'command' field
            parts = line.rsplit(',', 1)
            if len(parts) == 2:
                id_and_command, encrypted_cmd = parts
                # Split the ID from the command part
                id_str, command_str = id_and_command.split(',', 1)
                autocommands.append((int(id_str), command_str, encrypted_cmd.strip()))
    return pd.DataFrame(autocommands, columns=['ID', 'command', 'encrypted_cmd'])

# Global variable to store latest frame
latest_frame = None
lock = threading.Lock()

def display_frames():
    """Continuously displays the latest frame in parallel."""
    global latest_frame

    while True:
        with lock:
            if latest_frame is not None:
                cv2.imshow("CGReplay Demo: Live Game Video Stream", latest_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        time.sleep(0.01)  # Small delay to reduce CPU usage

# Start display thread
if live_watching == True:
    display_thread = threading.Thread(target=display_frames, daemon=True)
    display_thread.start()

# Load autocommand.txt
sync_df = load_syncfile(sync_file)

# kill all ports
subprocess.run("../port_clean.sh")

print(f"palyer is ready to receive {player_port} & command sent on {my_command_port}")

# Function to send command to server
def send_command(frame_id, encrypted_cmd, interface_name= player_interface, type='command', number = 0, fps = 0, cps = 0): # #"enp0s31f6" wlp0s20f3
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    #sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, interface_name.encode()) #interface_name.encode()) #interface_name.encode())
    sock.bind((player_ip, player_port))   # player IP + player port to receive the video 
    timestamp = time.perf_counter() #time.time() * 1000
    message = f"{timestamp},{encrypted_cmd},{frame_id},{type},{number},{fps},{cps}"
    # port setup
    #my_test_port = 5555
    sock.sendto(message.encode(),(cg_server_ip, my_command_port))
    #print("***"+player_interface+"***")
    
    sock.close()

# Function to read the QR code from the frame
def read_qr_code_from_frame(frame):
    """Reads the QR code from a given frame and extracts its data."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    qr_codes = pyzbar.decode(blurred_frame)

    for qr in qr_codes:
        qr_data = qr.data.decode('utf-8')
        print(f"Detected QR Code Data: {qr_data}")
        data_parts = qr_data.split(',')
        frame_id = None
        for part in data_parts:
            if "ID:" in part:
                frame_id = part.split(':')[1].strip()
                break
        if frame_id:
            return int(frame_id), qr_data

    return None, None



if scream_state==False:
    # GStreamer pipeline to receive video stream from port 5000
    gstreamer_pipeline = (
         f"udpsrc port={player_port} ! application/x-rtp, payload=96 ! "
        "queue max-size-time=1000000000 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink"
    )
else:
    # Run receiver.sh and capture the pipeline output
    receiver_output = subprocess.run([scream_receiver], capture_output=True, text=True, shell=True)
    gstreamer_pipeline = receiver_output.stdout.strip()  # Remove any extra whitespace
    print(f"Using GStreamer pipeline: {gstreamer_pipeline}")

# Open the video stream using OpenCV and GStreamer
cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()


# frame_buffer = deque(maxlen=30)  # Buffer to store frames
frame_counter = 1
#timeout_duration = 0.0001
previous_command = None
next_frame = 1
cmd_previoustime =frm_previoustime = time.perf_counter()
currrent_cps = 0
current_fps = 0
my_try_counter = 0 

while True:

    start_time = time.perf_counter() # time.time()

    # Try to receive the next frame
    ret, frame = cap.read()
    frm_rcv = time.perf_counter() # time.time() * 1000
    #test_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
    #print("Debug:***************",test_timestamp)

    # Read QR code from the buffered frame
    frame_id, qr_data = read_qr_code_from_frame(frame)
    current_fps = 1/(frm_rcv-frm_previoustime)
    frm_previoustime = frm_rcv
    #print(f"{frame_id}-fps:{current_fps}")
    
    # set the display thread!
    with lock:
        latest_frame = frame.copy()  # Update frame for live display

    if frame_id:
        print(f"Detected Frame ID: {frame_id}")
        
        if (my_try_counter%ack_freq)==0:
            send_command(frame_id,current_fps,player_interface,type='Ack', fps = current_fps, cps = currrent_cps )
        else:
            pass

        #next_frame = int(frame_id) + 1

        if frame_id == frame_counter+1:
            frame_filename = f"{received_frames}/{frame_id:04d}_{frm_rcv}.png"
        else:
            frame_filename = f"{received_frames}/{frame_id:04d}_{frm_rcv}_retry.png"

        frame_counter = frame_id
        
    else:
        print("No QR code detected in this frame.")
        send_command(0,"Downgrade",type='Nack',fps = current_fps, cps = currrent_cps )   # Send NacK
        send_command(frame_counter, previous_command,type='command',fps = current_fps, cps = currrent_cps ) # Send the Previous Command
        #continue
        #frame_counter+=1
        frame_filename = f"{received_frames}/{frame_counter:04d}_{frm_rcv}_NoQR.png"
        pass 
    
    
    # Save the current frame to a file
    cv2.imwrite(frame_filename, frame)
    #print(f"Saved {frame_filename}") /// Commented

   

    matching_command = [] 
    # Check if there's a matching command for this frame
    matching_command = sync_df[sync_df['ID'] == frame_counter]
    cmd_number = matching_command.shape[0]
    encrypted_cmds = matching_command['encrypted_cmd'].values

    print('\n********************************\n')
    if not matching_command.empty:
        #print(f"Match found for Frame {frame_counter}")
        
        send_command(frame_counter, encrypted_cmds,type ='command', number = cmd_number, fps = current_fps, cps= currrent_cps)
        cmd_sent = time.perf_counter() # time.time() * 1000
        currrent_cps = 1/(cmd_sent - cmd_previoustime)
        cmd_previoustime = cmd_sent
        #matching_command.apply(lambda row: send_command(frame_counter, encrypted_cmds,number = cmd_number), axis=1)  #row['encrypted_cmd'],number = cmd_number), axis=1)
        previous_command = encrypted_cmds.copy() # matching_command.iloc[0]['encrypted_cmd']
        
            # Log frame received time
        with open(rate_log, "a") as f: # fID - fps - cps
            f.write(f"{frame_id},{current_fps},{currrent_cps}\n")


        with open(time_log, "a") as f: # FID - F timestamp - CMD Timestamp
            f.write(f"{frame_id},{frm_rcv},{cmd_sent}\n")


    my_try_counter = my_try_counter + 1
    print(f'Recieved Frame # is: {my_try_counter}')
    if my_try_counter == stop_frm_number:
         break

    # Press 'q' to exit the video display window
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

cap.release()
cv2.destroyAllWindows()
