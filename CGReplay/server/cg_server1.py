'''
# Date: 2025-03-10
# Lab: LERIS/UFSCar
# Author: Alireza Shirmarz
# This is Configured for Netsoft 2025 Conference!
'''

import os , sys, time, socket, select, subprocess, cv2, qrcode, hashlib, yaml
import numpy as np, pandas as pd

os.sched_setaffinity(0, {0})

# Load configuration from YAML file
with open("../config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load settings from YAML file
# Loading Running Setup ****************************************************************************************
stop_frm_number = config["Running"]["stop_frm_number"]
game_name = config["Running"]["game"]

# Loading CG Server Setup*************************************************************************************** 
cg_server_port = config["server"]["server_port"]    # Port for receiving control (Joystick) commands from player
cg_server_ipadress = config["server"]["server_IP"]  # CG Server IP address
## Log Files Address in the CG Server
rate_control_log = config["server"]["log_rate_control"]     # Logging the Encoding (H.264) Rate Control 
server_log = config["server"]["log_server"]                 # Logging the Rate, FPS, CMD Rate .. Main Log
# frame_id,received_fame_id,my_gap,received_time,send_time,current_srv_fps,received_fps,current_cps,received_cps,current_srv_fps/received_fps,received_cps/current_cps,bitrate
frame_log = config["server"]["log_frame"]                   # Logging Frame ID, current_srv_fps, processing_time, bitrate 

'''
Referesh Logs
'''
# Remove rate_Control log
if os.path.exists(rate_control_log):
    os.remove(rate_control_log)

with open(rate_control_log, "w") as f:
    f.write("frame_id,rate_ctl\n")


# Remove server log
if os.path.exists(server_log):
    os.remove(server_log)

with open(server_log, "w") as f:
    f.write("frame_id,received_fame_id,my_gap,received_time,send_time,current_srv_fps,received_fps,current_cps,received_cps,current_srv_fps/received_fps,received_cps/current_cps,bitrate\n")



# Remove frame Log
if os.path.exists(frame_log):
    os.remove(frame_log)

with open(frame_log, "w") as f:
    f.write("frame_id,resolution,Frame Size(Byte),GOP,current_srv_fps,processing_time,bitrate\n")




# All frames with Bitrate
cg_server_socket_timeout = config["server"]["socket_timeout"]

# Loading CG Gamer or Player Setup****************************************************************************** 
player_ip = config['gamer']["player_IP"]                     # CG Gamer IP address
player_port = config['gamer']["player_streaming_port"]       # UDP Port for streaming video to Gamer
my_command_port = config['gamer']["player_command_port"]     # UDP Port for receiving the command in the CG server


# Loading CG Server Sync file and Frames ***********************************************************************
folder_path = config[game_name]["frames"]                     # The folder includes the frames in png format!
my_command_frame_addr = config[game_name]["sync_file"]        # The sync file for scynching between frames and commands!

# Loading Encoding Setup ***************************************************************************************
fps = config["encoding"]["fps"]                                 # Frame Rate (fps)
resolution_width = config["encoding"]["resolution"]["width"]    # Width 
resolution_height = config["encoding"]["resolution"]["height"]  # Height
GOP = config["encoding"]["GOP"]
MyvideoEncoder = config["encoding"]["name"] # Encoder name e.g., H.264/H.265
myencoder = config["encoding"][MyvideoEncoder]["encoder"]
myparser = config["encoding"][MyvideoEncoder]["parsing"]
myrtp = config["encoding"][MyvideoEncoder]["packetization"]
# x265enc emits VPS/SPS/PPS only once by default. The player launches ~1s after
# the server, so it joins mid-stream and never sees the parameter sets — it
# decodes nothing. repeat-headers=1 puts them on every IDR (x264enc already
# repeats SPS/PPS, so H.264 needs no extra option).
enc_extra = 'option-string="repeat-headers=1"' if MyvideoEncoder in ("H.265", "H265", "HEVC") else ""
# Loading Protocols Setup **************************************************************************************
scream_state=config["protocols"]["SCReAM"]                      # CCA Protocol for UDP as SCReAM developed by Ericsson!
scream_sender=config["protocols"]["sender"]                     # Sender as CGServer!
quic_state  =config["protocols"].get("QUIC", False)            # QUIC transport (replaces RTP when True)
roq_state   =config["protocols"].get("RoQ", False)             # RoQ transport: RTP over QUIC datagrams


# Loading Sync Setup *******************************************************************************************
Enc_Rate_jump = config["sync"]["jump"]                          # CGReplay Encoding Frame Rate jump!
Enc_Rate_rise = config["sync"]["rise"]                          # CGReplay Encoding Frame Rate rise!
Enc_Rate_decrease = config["sync"]["decrease"]                  # CGReplay Encoding Frame Rate decrease!
Enc_Rate_fall = config["sync"]["fall"]                          # CGReplay Encoding Frame Rate fall!


import struct

def parse_rtcp_app_packet(data):
    # RTCP header: 1 byte v_p_rc, 1 byte pt, 2 bytes length
    # 4 bytes name, then payload
    if len(data) < 8:
        return None  # Not a valid RTCP APP packet
    v_p_rc, pt, length = struct.unpack("!BBH", data[:4])
    name = data[4:8]
    payload = data[8:]
    # Optionally, check pt == 204 (APP)
    return payload.decode()



def get_pipeline_str(frame_dir):
    """Runs send.sh and captures only the GStreamer pipeline string."""
    try:
        # Run send.sh and capture the output
        result = subprocess.run(
            ["/home/alireza/scream/scream/gstscream/scripts/sender3.sh", frame_dir ], 
            capture_output=True, 
            text=True,
            check=True
        )

        # Split the output into lines
        output_lines = result.stdout.split("\n")

        # Find the first line that contains 'rtpbin' (start of a GStreamer pipeline)
        for line in output_lines:
            if "rtpbin" in line:
                return line.strip()  # Return only the GStreamer pipeline string

        print("Error: No valid GStreamer pipeline found in send.sh output!")
        return ""

    except subprocess.CalledProcessError as e:
        print(f"Error running send.sh: {e}")
        return ""


# Hash Function to encrypt the Commands
def hash_string(input_string, output_size):
    """Hashes a string using SHAKE and returns the hex digest with the given output size in bytes."""
    # Use SHAKE-128 for flexibility in output size
    shake = hashlib.shake_128()
    shake.update(input_string.encode('utf-8'))
    # Return the hex digest with the specified byte size
    return shake.hexdigest(output_size)

# Custom function to load autocommands.txt while handling the complex 'command' field
def load_syncfile(file_path):
    synccommand = []
    with open(file_path, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            # Split only on the last comma to avoid splitting inside the 'command' field
            parts = line.rsplit(',', 1)
            if len(parts) == 2:
                id_and_command, encrypted_cmd = parts
                # Split the ID from the command part
                id_str, command_str = id_and_command.split(',', 1)
                synccommand.append((int(id_str), command_str, encrypted_cmd.strip()))
    return pd.DataFrame(synccommand, columns=['ID', 'command', 'encrypted_cmd'])

# List of frame IDs where we want to pause and wait for socket input
sync_df = load_syncfile(my_command_frame_addr) # Load autocommands for fram/command ordering
pause_frame_ids = sync_df['ID'].tolist()
# Backup for Forza
resolution = (resolution_width,resolution_height)

def generate_qr_code(data):
    """Generate QR code as an image from the given data."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size= 20, #10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)

    # Create an image from the QR Code instance
    qr_img = qr.make_image(fill='black', back_color='white')
    
    # Convert to numpy array for OpenCV compatibility
    qr_img = np.array(qr_img.convert('RGB'))
    
    return qr_img

# Setup Socket for Receiving the Commands
def setup_socket():
    """Set up a UDP socket to send control messages."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Allow reuse of the same address and port
    #sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

    # Bind the socket to the specific source port (same as GStreamer)
    sock.bind((cg_server_ipadress, my_command_port)) # cg_server_port))  # It can be  the same Streaming or differnt!
    print(f"Listening on UDP port {cg_server_port} for streaming & {my_command_port} for control (Joystick) data...")
    return sock

# Stream the video frames
received_fame_id = 0

def stream_frames(game_name):
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    Gst.init(None)

    global bitrate
    # setup Encoding H.264
    bitrate = config["encoding"]["starting_bitrate"]
    bitrate_min = config["encoding"]["bitrate_min"] # 2000
    bitrate_max = config["encoding"]["bitrate_max"] # 8000

    # setup Synchronization sliding window min & max
    window_min = config["sync"]["window_min"] # 1 
    window_max = config["sync"]["window_max"] # 4
 

    cmd_previous_time =  time.perf_counter()
    
    # bitrate = 10000  # in kbps
    """Stream frames with QR code embedded over UDP using GStreamer."""
    # Set up the control socket
    control_socket = setup_socket()
    #control_socket.settimeout(1)

    if scream_state == False:
        ''' The main which worked!'''
        print(f"It's value is {scream_state}-SCReAM Disabled!")
        # Non-Scream
        '''
        pipeline_str = f"""
            appsrc name=source is-live=true block=true format=GST_FORMAT_TIME do-timestamp=true !
            videoconvert ! video/x-raw,format=I420,width={resolution_width},height={resolution_height},framerate={fps}/1 !
            x264enc bitrate={bitrate} speed-preset=ultrafast tune=zerolatency key-int-max={GOP} !
            h264parse ! rtph264pay ! 
            udpsink host={player_ip} port={player_port} bind-port={cg_server_port}
        """
        '''

        pipeline_str = f"""
            appsrc name=source is-live=true block=true format=GST_FORMAT_TIME do-timestamp=true !
            videoconvert ! video/x-raw,format=I420,width={resolution_width},height={resolution_height},framerate={fps}/1 !
            {myencoder} bitrate={bitrate} speed-preset=ultrafast tune=zerolatency key-int-max={GOP} {enc_extra} !
            {myparser} ! {myrtp} !
            udpsink host={player_ip} port={player_port} bind-port={cg_server_port}
        """
        
        # frame_size_bytes ≈ (bitrate_kbps * 1000) / 8 / fps

        pipeline = Gst.parse_launch(pipeline_str)
        #print(pipeline)
        appsrc = pipeline.get_by_name("source")
    else:
        print(f"It's value is  {scream_state}-SCReAM Enabled")
        # SCReAM
        
        sender_output = subprocess.run([scream_sender], capture_output=True, text=True, shell=True)
        pipeline_str = sender_output.stdout.strip()
        if not pipeline_str:
            print("ERROR: sender.sh returned empty pipeline string.")
            return
        print(f"Using GStreamer pipeline: {pipeline_str}")
        pipeline = Gst.parse_launch(pipeline_str)
        appsrc = pipeline.get_by_name("source")
    print("=========================")
    print(f"\n🔍 Debugging: Generated GStreamer Pipeline String:\n{pipeline_str}\n")
    print("=========================")
    # Parse and create the GStreamer pipeline
    #pipeline = Gst.parse_launch(pipeline_str)

    # Get the 'appsrc' element from the pipeline
    #appsrc = pipeline.get_by_name("source")

    if not appsrc:
        print("Failed to retrieve appsrc element from the pipeline.")
        return

    # Set the caps for the 'appsrc' element, including FPS and resolution
    appsrc.set_property("caps", Gst.Caps.from_string(f"video/x-raw,format=BGR,width={resolution_width},height={resolution_height},framerate={fps}/1"))

    # Start the pipeline
    pipeline.set_state(Gst.State.PLAYING)
    
    #frame_id = 1  # Frame counter (starting from 1 for human-readable frame IDs)
    png_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])  # List PNG files
    # flag_lock = False
    cmd_counter = 0  #sync_df.shape[0] # max number 
    previous_time = time.perf_counter()
    

    idx = 0
    while idx < len(png_files):
        #if idx == stop_frm_number: # stop after streaming 'stop_frm_number' frames! 
            #break
        file = png_files[idx]  # Access the file by index
        print("Log hint==>>>",file, idx)
        
        # Note: Checkpoint

        # Construct the full file path
        frame_path = os.path.join(folder_path, file)

        # Load the frame
        frame = cv2.imread(frame_path)  # Read the image file
        frame_id = int(file.split('.')[0])  # Extract frame ID from the filename
        print("Debug:==>", frame_id)

        if frame_id == stop_frm_number: # stop after streaming 'stop_frm_number' frames! 
            break

        if frame is None:
            print(f"Could not load frame {file}")
            idx += 1  # Move to the next file if loading fails
            continue    
        
        #idx= idx + 1


        
        timestamp = time.perf_counter()
        # Resize frame to the desired resolution
        #frame = cv2.resize(frame, resolution)
        frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)


        qr_data = f"Frame ID: {frame_id}, rcv_timestamp: {timestamp}, resolution: {resolution},bitrate:{bitrate}"
        qr_img = generate_qr_code(qr_data)
        
        qr_size = 200 #100  # QR code size in pixels
        qr_img = cv2.resize(qr_img, (qr_size, qr_size))

        # Overlay the QR code onto the bottom-right corner of the frame
        x_offset = frame.shape[1] - qr_size - 10  # 10px padding from the right edge
        y_offset = frame.shape[0] - qr_size - 10  # 10px padding from the bottom edge
        frame[y_offset:y_offset + qr_size, x_offset:x_offset + qr_size] = qr_img

        # Convert frame to bytes and push to GStreamer
        frame_byte = frame.tobytes()
        gst_buffer = Gst.Buffer.new_wrapped(frame_byte) #frame.tobytes())
         
        # Debug: Keep it in mind 
        appsrc.emit("push-buffer", gst_buffer)
        # fpscomputing + processing time (Rendering)
        #############################################################################################################
        # Note: This code was added because the first frame was not being sent properly! 
        if frame_id==1:
            #gst_buffer = Gst.Buffer.new_wrapped(frame_byte) #frame.tobytes())
            appsrc.emit("push-buffer", gst_buffer)
            print('Sending the frame one again')
        #############################################################################################################

        my_fps_time = time.perf_counter() 
        current_srv_fps = 1/(my_fps_time - previous_time)
        processing_time = my_fps_time - timestamp
        previous_time = my_fps_time

        # Log the frame that is being streamed (frame_log.txt)
        print(f"Streaming frame {frame_id}")
        with open(frame_log, "a") as f: f.write(f"{frame_id},{resolution},{len(frame_byte)},{GOP},{current_srv_fps},{processing_time},{bitrate}\n")
        
        # Socket Time out 
        timeout = cg_server_socket_timeout ## socket timeout is set in config file!
        #timeout = 0.0001 # if not flag_lock else None
        ready_to_read, _, _ = select.select([control_socket], [], [], timeout) #0.01)  # 10 milliseconds timeout         
        received_fame_id = 0 

        if ready_to_read:
            #print('It is ready ready to receive!!!!!!')
            # Receive control data for each matching command (blocking)

            data, addr = control_socket.recvfrom(1024)
            received_data = data.decode().split(',')
            received_time = time.perf_counter() #time.time() * 1000  # Timestamp in milliseconds
            current_cps = 1/(received_time - cmd_previous_time)
            cmd_previous_time = received_time

            send_time = received_data[0]
            received_cmd = received_data[1]
            received_fame_id = int(received_data[2])
            received_type = received_data[3]
            received_cmd_number = received_data[4]
            received_fps = float(received_data[5])
            received_cps =  float(received_data[6])
            #print(received_type)
            print(f"Received control data: Type ({received_type}) Time = {send_time}, from {addr}")
            #print(f"Debug: {received_cmd} | Number: {cmd_number}")






            my_gap = max((frame_id - received_fame_id),0) # to check the window 
            Nack_counter = 0
            rate_ctl = [None , None, None, None]


            """
            Logging the received data (srv_QoEMetrics) 
            """
            with open(server_log, "a") as f: 
                f.write(f"{frame_id},{received_fame_id},{my_gap},{received_time},{send_time},{current_srv_fps},{received_fps},{current_cps},{received_cps},{current_srv_fps/received_fps},{received_cps/current_cps},{bitrate} \n")
                print("Total is logging!")



            if received_type=='Ack':
                rate_ctl = [None , None, None, None]
                rate_ctl[3] = 'Ack'
                Nack_counter = 0 
                if my_gap <= window_min:    # Check to keep sync using sliding between min/max window
                    print(f'(Ack) [High Sync:(Fast Rate Increase)] ==> Frame ID is {frame_id} with Gap {my_gap} \n [player fps = {received_fps}] [server fps = {current_srv_fps}] \n [player cps = {received_cps} [server cps = {current_cps}]] | bitrate = {bitrate}')
                    bitrate = bitrate + (bitrate * Enc_Rate_jump) if bitrate_min <= bitrate <= bitrate_max else bitrate
                    #rate_ctl[0] = 'Fast Increase', rate_ctl[1] = 0.2, rate_ctl[2] = bitrate
                    rate_ctl = ['Rate Jump', Enc_Rate_jump, bitrate,rate_ctl[3]]

                elif window_min < my_gap <= window_max:
                    print(f'(Ack) [Sync: (Rate Increase)] ==> Frame ID is {frame_id} with Gap {my_gap} \n [player fps = {received_fps}] [server fps = {current_srv_fps}] \n [player cps = {received_cps} [server cps = {current_cps}]] | bitrate = {bitrate}')
                    bitrate = bitrate + (bitrate * Enc_Rate_rise) if bitrate_min <= bitrate <= bitrate_max else bitrate
                    rate_ctl = ['Rate Rise', Enc_Rate_rise,bitrate,rate_ctl[3]]
                    
                elif my_gap > window_max:
                    print(f'(Ack) [Critical Sync: (Rate Decrease)] ==> Frame ID is {frame_id} with Gap {my_gap} \n [player fps = {received_fps}] [server fps = {current_srv_fps}] \n [player cps = {received_cps} [server cps = {current_cps}]] | bitrate = {bitrate}')
                    # print(f'(Ack) (**Wait**) ==> Received Frame is {received_fame_id} == current {frame_id} \n [fps = {received_fps}] [server fps = {current_srv_fps}] but Gap is {my_gap}')
                    bitrate = bitrate - (bitrate * Enc_Rate_decrease) if bitrate_min <= bitrate <= bitrate_max else bitrate
                    rate_ctl = ['Rate Decrease', Enc_Rate_decrease, bitrate,rate_ctl[3]]
                    #idx = idx - my_gap

                #with open(rate_control_log, "a") as f: f.write(f"{frame_id},{rate_ctl}\n")

            elif received_type=='Nack':
                rate_ctl = [None, None, None, None]
                rate_ctl[3] = 'command'
                if Nack_counter == 0:
                    #print(f'(Nack) ==> Received Frame is {received_fame_id}')
                    print(f'(Nack) [Not Sync: (Decrease & lagged!)] ==> Frame ID is {frame_id} with Gap {my_gap} \n [player fps = {received_fps}] [server fps = {current_srv_fps}] \n [player cps = {received_cps} [server cps = {current_cps}]] | bitrate = {bitrate}')
                    bitrate = bitrate - (bitrate * 0.2) if bitrate_min <= bitrate <= bitrate_max else bitrate
                    idx = received_fame_id - round(my_gap/2,) # Create the lag!  
                    #time.sleep(0.0001)
                    Nack_counter = Nack_counter + 1
                    rate_ctl = ['Rate Fall & lagged', [0.2,1], bitrate,rate_ctl[3]]
                    

                else:
                    print(f'(Nack) [Fast Decrease & lagged!] ==> Frame ID is {frame_id} with Gap {my_gap} \n [player fps = {received_fps}] [server fps = {current_srv_fps}] \n [player cps = {received_cps} [server cps = {current_cps}]] | bitrate = {bitrate}')
                    bitrate = bitrate - (bitrate * 0.5) if bitrate_min <= bitrate <= bitrate_max else bitrate
                    idx = received_fame_id - (my_gap/2)
                    #time.sleep(0.0001)
                    Nack_counter = Nack_counter + 1
                    rate_ctl = ['Fast Decrease & lagged', [0.5,my_gap], bitrate,rate_ctl[3]]
                #with open(rate_control_log, "a") as f: f.write(f"{frame_id},{rate_ctl}\n")


            elif received_type=='command':
                rate_ctl = [None, None, None, None]
                rate_ctl[3] = 'command'

                matching_commands = []
                matching_commands = sync_df[sync_df['ID'] == (received_fame_id)] #received_fame_id] #expected_command_id]
                my_cmd_number = matching_commands.shape[0]
                cmd_counter = cmd_counter + my_cmd_number


                if not matching_commands.empty:
                    state = [None , None]
                    if pause_frame_ids[cmd_counter-1] == matching_commands['ID'].iloc[0]:
                        state[0] ='Sync'
                        print(f"Sync***{my_gap}")
                    else:
                        print(f"Ooops***{my_gap}")
                        state[0] = 'Not Sync'
                    
                    if my_gap <= window_min:
                        bitrate = bitrate + (bitrate * Enc_Rate_jump) if bitrate_min <= bitrate <= bitrate_max else bitrate
                        state[1] = 'Rate Jump'
                        rate_ctl = ['Rate Jump',  Enc_Rate_jump,  bitrate,rate_ctl[3]]

                    elif window_min < my_gap <= window_max:
                        bitrate = bitrate + (bitrate * Enc_Rate_rise) if bitrate_min <= bitrate <= bitrate_max else bitrate
                        state[1] = 'Rate Rise'
                        rate_ctl = ['Rate Rise', Enc_Rate_rise,  bitrate,rate_ctl[3]]

                    elif my_gap > window_max:
                        bitrate = bitrate - (bitrate * Enc_Rate_fall) if bitrate_min <= bitrate <= bitrate_max else bitrate
                        state[1] = 'Rate Fall'
                        #idx = idx - my_gap
                        rate_ctl = ['Fast Decrease & lagged',[Enc_Rate_fall, my_gap],  bitrate,rate_ctl[3]]
                        continue
                    
                    print(f"{state} | Gap:{my_gap} | FID:{frame_id} | player FID:{received_fame_id}"
                        f" player fps = {received_fps} | server fps = {current_srv_fps}"
                        f" player cps = {received_cps} | server cps = {current_cps}, , rate = {bitrate} ") 
                    

                    #with open(server_log, "a") as f: 
                        #f.write(f"{frame_id},{received_fame_id},{my_gap},{received_time},{send_time},{current_srv_fps},{received_fps},{current_cps},{received_cps},{current_srv_fps/received_fps},{received_cps/current_cps},{bitrate} \n")
                        #print("Total is logging!")
            
                    state = [None , None]

            with open(rate_control_log, "a") as f: f.write(f"{frame_id},{rate_ctl}\n")        
        idx= idx + 1
        
            
    # End the stream
    appsrc.emit("end-of-stream")
    pipeline.set_state(Gst.State.NULL)

def load_config(file_path="config.txt"):
    """Reads the config.txt file and returns a dictionary of settings."""
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Skip empty lines and comments
            if line.strip() and not line.strip().startswith("#"):
                key, value = line.split("=")
                key = key.strip()
                value = value.strip()
                # Parse resolution into a tuple
                if key == "resolution":
                    value = tuple(map(int, value.split(",")))
                elif key in ["fps", "bitrate", "player_port", "my_test_port", "cg_server_port"]:
                    value = int(value)  # Convert numeric values
                config[key] = value
    return config

if __name__ == "__main__":
    print(f'Started streaming to {cg_server_port}... ')
    if quic_state:
        from quic_sender import main as quic_main
        import asyncio
        print("[CGReplay] QUIC transport selected — delegating to quic_sender")
        asyncio.run(quic_main())
    elif roq_state:
        from roq_sender import main as roq_main
        import asyncio
        print("[CGReplay] RoQ transport selected — delegating to roq_sender")
        asyncio.run(roq_main())
    else:
        stream_frames(game_name)

    
