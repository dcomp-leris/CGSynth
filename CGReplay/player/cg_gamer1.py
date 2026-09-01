'''
# Date: 2024-11-29
# Author: Alireza Shirmarz
# Lab: LERIS/UFScar 
# This is Configured for Netsoft 2025 Conference!
# Gamer: (1) 
'''

import cv2, os, socket, time, yaml, threading, subprocess, glob
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
capturing_options = config.get("capturing_options", {})
enable_pcap = capturing_options.get("enable_pcap", False)
pcap_file = config["gamer"].get("pcap_file")


# server setup
cg_server_ip = config["server"]["server_IP"]        # CG Server IP address
cg_server_port = config["server"]["server_port"]    # Port for receiving control (Joystick) commands from player

# client (player) setup
player_ip = config['gamer']["player_IP"]                     # CG Gamer IP address
player_port =config['gamer']["player_streaming_port"]       # UDP Port for streaming video to Gamer
my_command_port = config['gamer']["player_command_port"]

# sync setup
folder_path = config[game_name]["frames"] 
sync_file = config[game_name]["sync_file"]  

# log setup 
rate_log = config["gamer"]["player_rate_log"] 
time_log = config["gamer"]["player_time_log"]
frame_log = config["gamer"]["player_frame_log"]
received_frames = config["gamer"]["received_frames"]

# Ensure player logs directory exists
logs_dir = os.path.dirname(rate_log)
if logs_dir:
    os.makedirs(logs_dir, exist_ok=True)


'''
Referesh Logs
'''

# Ensure received_frames directory exists and clean it
if not os.path.exists(received_frames):
    os.makedirs(received_frames)
elif os.path.exists(received_frames):
    [os.remove(f) for f in glob.glob(received_frames+"/*") if os.path.isfile(f)]

# Remove rate_Control log
if os.path.exists(rate_log):
    os.remove(rate_log)

# Remove server log
if os.path.exists(time_log):
    os.remove(time_log)

# QUIC path — delegate BEFORE touching the RTP/SCReAM logs below. Otherwise a
# QUIC run truncates ply_frame.csv / responsetime_CG.csv and wipes the data the
# other two modes wrote on their own runs.
quic_state = config["protocols"].get("QUIC", False)
if quic_state:
    subprocess.run("../port_clean.sh")
    with open("/tmp/player_ready", "w") as f:
        f.write("ready")
    from quic_receiver import run_player as quic_run
    print("[CGReplay] QUIC transport selected — delegating to quic_receiver", flush=True)
    quic_run()
    raise SystemExit(0)

# RoQ path — same rationale as QUIC: delegate before touching the shared logs.
roq_state = config["protocols"].get("RoQ", False)
if roq_state:
    subprocess.run("../port_clean.sh")
    with open("/tmp/player_ready", "w") as f:
        f.write("ready")
    from roq_receiver import run_player as roq_run
    print("[CGReplay] RoQ transport selected — delegating to roq_receiver", flush=True)
    roq_run()
    raise SystemExit(0)

# Remove frame Log
if os.path.exists(frame_log):
    os.remove(frame_log)

# Create new logs with headers
with open(rate_log, "w") as f:
    f.write("frame_id,fps,cps\n")
with open(time_log, "w") as f:
    f.write("frame_id,frame_timestamp,cmd_timestamp\n")
with open(frame_log, "w") as f:
    f.write("frame_id,frame_counter,fps,retry_status\n")



player_interface = config["gamer"]["player_interface"]

# Do you want to watch the Game Video live? 
live_watching = config["Running"]["live_watching"]



# Ack Rate
ack_freq = config["sync"]["ack_freq"]

# Encoding Setup
MyvideoEncoder = config["encoding"]["name"] # Encoder name e.g., H.264/H.265 
mydecoder = config["encoding"][MyvideoEncoder]["decoder"]
myrtp = config["encoding"][MyvideoEncoder]["Depacketization"]


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

# Add to synch for mininet!!
with open("/tmp/player_ready", "w") as f:
    f.write("ready")


print(f"player is ready to receive {player_port} & command sent on {my_command_port}")
if enable_pcap:
    print(f"PCAP capture is ENABLED; expected output path: {pcap_file}")
else:
    print("PCAP capture is DISABLED (capturing_options.enable_pcap=False)")

# Function to send command to server (Pure UDP)

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
'''
import struct
# Function to send command to server (RTP over UDP)
def send_command(frame_id, encrypted_cmd, interface_name=player_interface, type='command', number=0, fps=0, cps=0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    # Optionally bind to interface
    # sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, interface_name.encode())
    sock.bind((player_ip, player_port))

    timestamp = time.perf_counter()
    message = f"{timestamp},{encrypted_cmd},{frame_id},{type},{number},{fps},{cps}".encode()

    # Minimal RTCP header: Version(2 bits), Padding(1), RC(5), PT(8), length(16)
    # We'll use PT=204 (APP packet, for custom data)
    v_p_rc = (2 << 6) | 0  # Version 2, no padding, RC=0
    pt = 204  # APP packet
    length = (len(message) + 8) // 4 - 1  # RTCP length field is in 32-bit words minus one

    # RTCP header: 1 byte v_p_rc, 1 byte pt, 2 bytes length
    header = struct.pack("!BBH", v_p_rc, pt, length)
    # 4 bytes name (for APP packets, can be anything)
    name = b'CGPL'
    rtcp_packet = header + name + message

    sock.sendto(rtcp_packet, (cg_server_ip, my_command_port))
    sock.close()
'''
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

    return -1, None



if scream_state==False:
    # timeout=3000000000 ns = 3s — after server stops, udpsrc emits EOS so cap.read() returns False
    gstreamer_pipeline = (
        f"udpsrc port={player_port} timeout=3000000000 ! application/x-rtp, payload=96 ! "
        f"queue max-size-time=1000000000 ! {myrtp} ! {mydecoder} ! videoconvert ! appsink"
    )
    '''
    gstreamer_pipeline = (
         f"udpsrc port={player_port} ! application/x-rtp, payload=96 ! "
        "queue max-size-time=1000000000 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink"
    )'''
else:
    # Run receiver.sh and capture the pipeline output
    receiver_output = subprocess.run([scream_receiver], capture_output=True, text=True, shell=True)
    gstreamer_pipeline = receiver_output.stdout.strip()  # Remove any extra whitespace
    print(f"Using GStreamer pipeline: {gstreamer_pipeline}")

# Watchdog: arm it BEFORE opening the capture. VideoCapture creation and the
# first cap.read() can block forever if the server never delivers a decodable
# frame (e.g. a stall), so the watchdog must already be running by then.
#
# It is IDLE-based, not a fixed total duration: as long as frames keep arriving
# the player keeps running, so slow encoders (H.265) and rate-limited transports
# (SCReAM) can stream all the way to the end instead of being cut off part-way.
# A fixed-duration watchdog cut the tail of every slow run (frames after the
# cutoff were simply never received). This exits only after _IDLE_TIMEOUT seconds
# with no new frame (true end-of-stream or a genuine stall); _HARD_CAP is a
# last-resort absolute ceiling so it can never hang forever.
import threading as _threading, os as _os
_IDLE_TIMEOUT = 20.0  # seconds with no new frame → assume the stream ended
_HARD_CAP     = (stop_frm_number / config["encoding"]["fps"]) * 12 + 60
_activity     = [time.perf_counter()]   # updated in the receive loop per frame
_start_ts     = time.perf_counter()
def _watchdog():
    while True:
        _threading.Event().wait(timeout=2.0)
        now = time.perf_counter()
        if (now - _activity[0]) > _IDLE_TIMEOUT or (now - _start_ts) > _HARD_CAP:
            print("RTP streaming complete.")
            _os._exit(0)
_threading.Thread(target=_watchdog, daemon=True).start()

# Open the video stream using OpenCV and GStreamer
cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ ERROR: Could not open video stream")
    print("\nDETAILED DIAGNOSTICS:")
    print("-" * 30)
    
    # Try to get more info about the failure
    print("Possible causes:")
    print("1. ❌ OpenCV not compiled with GStreamer support")
    print("2. ❌ GStreamer plugins missing")
    print("3. ❌ Network/port issues")
    print("4. ❌ No video stream available on specified port")
    print("5. ❌ Pipeline syntax errors")
    
    print(f"\nTROUBLESHOOTING STEPS:")
    print("-" * 30)
    print("1. Test if video stream is available:")
    print(f"   gst-launch-1.0 udpsrc port={player_port} ! fakesink dump=true")
    
    print("\n2. Test pipeline manually:")
    print(f"   gst-launch-1.0 {gstreamer_pipeline.replace('appsink', 'autovideosink')}")
    
    print("\n3. Check network connectivity:")
    print(f"   netstat -ulnp | grep {player_port}")
    
    print("\n4. Install missing GStreamer plugins:")
    print("   sudo apt-get install gstreamer1.0-plugins-base gstreamer1.0-plugins-good")
    print("   sudo apt-get install gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly")
    
    print("\n5. Rebuild OpenCV with GStreamer:")
    print("   pip uninstall opencv-python")
    print("   pip install opencv-contrib-python")

    
    cap.release()
    exit(1)
else:
    print("✅ SUCCESS: Video stream opened successfully!")
    
    # Test frame reading
    print("Testing frame capture...")
    ret, test_frame = cap.read()
    if ret and test_frame is not None:
        print(f"✅ Successfully captured test frame: {test_frame.shape}")
    else:
        print("⚠️  Warning: Could not read initial frame (stream might not be active yet)")
    
    print("\n" + "="*60)
    print("STARTING MAIN VIDEO PROCESSING LOOP")
    print("="*60)


# frame_buffer = deque(maxlen=30)  # Buffer to store frames
frame_counter = 1 #1
#timeout_duration = 0.0001
previous_command = None
#next_frame = 1
cmd_previoustime =frm_previoustime = time.perf_counter()
currrent_cps = 0
current_fps = 0
my_try_counter = 0

while True:

    start_time = time.perf_counter() # time.time()

    # Try to receive the next frame
    ret, frame = cap.read()
    frm_rcv = time.perf_counter()

    if not ret or frame is None:
        print("RTP streaming complete.")
        break

    _activity[0] = frm_rcv   # tell the idle watchdog a frame just arrived

    # Use frame_counter as frame_id (QR decoding skipped — too slow for RTP/SCReAM path)
    frame_id = frame_counter
    qr_data = None
    current_fps = 1/(frm_rcv-frm_previoustime)
    frm_previoustime = frm_rcv


    # logging buffer to make it faster
    log_frame_buffer = []
    buffer_size = 100  # Write every 100 frames
    previous_frame_id = 0
    # set the display thread!
    with lock:
        latest_frame = frame.copy()  # Update frame for live display

    if frame_id:
        print(f"Detected Frame ID: {frame_id}")
        #frame_counter = frame_id # Counter for No-QR Code frames
        


        if (my_try_counter%ack_freq)==0:
            send_command(frame_id,current_fps,player_interface,type='Ack', fps = current_fps, cps = currrent_cps)
            #log_frame_buffer.append(f"{frame_id},{current_fps},{0}\n")
            #if len(log_frame_buffer) >= buffer_size: open(frame_log, "a").writelines(log_frame_buffer); log_frame_buffer.clear()

            if frame_id == stop_frm_number:
                break
        else:

            pass
        
        
        #next_frame = int(frame_id) + 1

        frame_filename = f"{received_frames}/{frame_id:04d}.png"

        

        if frame_id!=frame_counter and (previous_frame_id+1)!=frame_id and frame_id!=1:
            print(f"⚠️  Frame ID Mismatch: Expected {frame_counter+1}, but got {frame_id}. Possible frame loss or out-of-order delivery.")
            # FID, FPS, Retry Status [noremal:0, retry:1, No_QR:2]
            log_frame_buffer.append(f"{frame_id},{frame_counter},{current_fps},{1}\n")
            if len(log_frame_buffer) >= buffer_size: open(frame_log, "a").writelines(log_frame_buffer); log_frame_buffer.clear()
            frame_counter = frame_id+1

        else:
            previous_frame_id = frame_id
            cv2.imwrite(frame_filename, frame)
            log_frame_buffer.append(f"{frame_id},{frame_counter},{current_fps},{0}\n")
            if len(log_frame_buffer) >= buffer_size: open(frame_log, "a").writelines(log_frame_buffer); log_frame_buffer.clear()
            frame_counter = frame_counter + 1
            


         
        '''
        if frame_id == frame_counter+1:
            #frame_filename = f"{received_frames}/{frame_id:04d}_{frm_rcv}.png"
            frame_filename = f"{received_frames}/{frame_id:04d}.png"
            ################################################################################## 
            # Save the current frame to a file
            #cv2.imwrite(frame_filename, frame)
            ##################################################################################

            # Write logs if buffer is full 
            # FID, FPS, Retry Status [noremal:0, retry:1, No_QR:2]
            log_frame_buffer.append(f"{frame_id},{current_fps},{0}\n")
            if len(log_frame_buffer) >= buffer_size: open(frame_log, "a").writelines(log_frame_buffer); log_frame_buffer.clear()
        else:
            #frame_filename = f"{received_frames}/{frame_id:04d}_{frm_rcv}_retry.png"
            frame_filename = f"{received_frames}/{frame_id:04d}_retry.png"
            # Write logs if buffer is full 
            # FID, FPS, Retry Status [noremal:0, retry:1, No_QR:2]
            log_frame_buffer.append(f"{frame_id},{current_fps},{1}\n")
            if len(log_frame_buffer) >= buffer_size: open(frame_log, "a").writelines(log_frame_buffer); log_frame_buffer.clear()
        '''

        
        # frame_counter = frame_id
        


        
    else:
        print("No QR code detected in this frame.")
        frame_counter = frame_counter + 1
        send_command(0,"Downgrade",type='Nack',fps = current_fps, cps = currrent_cps )   # Send NacK
        send_command(frame_counter, previous_command,type='command',fps = current_fps, cps = currrent_cps ) # Send the Previous Command
        #continue
        #frame_counter+=1
        #frame_counter = frame_counter + 1
        frame_filename = f"{received_frames}/{frame_counter:04d}_NoQR.png"
        # Write logs if buffer is full 
        # FID, FPS, Retry Status [noremal:0, retry:1, No_QR:2]
        #log_frame_buffer.append(f"{frame_id},C,{current_fps},{2}\n")
        #if len(log_frame_buffer) >= buffer_size: open(frame_log, "a").writelines(log_frame_buffer); log_frame_buffer.clear()



        # FID, FPS, Retry Status [noremal:0, retry:1, No_QR:2]
        log_frame_buffer.append(f"{frame_id},{frame_counter},{current_fps},{2}\n")
        if len(log_frame_buffer) >= buffer_size: open(frame_log, "a").writelines(log_frame_buffer); log_frame_buffer.clear()
        #pass
        #frame_counter = frame_counter + 1
    
    # Save the current frame to a file ############### Noting: Commented temporary!
    #cv2.imwrite(frame_filename, frame)
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
    print(f'Received Frame # is: {my_try_counter}')

    if my_try_counter >= stop_frm_number:
        print("RTP streaming complete.")
        break

    # Write any remaining in the log
    if log_frame_buffer:
        with open(frame_log, "a") as f:
            f.writelines(log_frame_buffer)


    #if my_try_counter == stop_frm_number or (max((frame_id),0)+1) == stop_frm_number:
         #break

    # Press 'q' to exit the video display window
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

cap.release()
cv2.destroyAllWindows()