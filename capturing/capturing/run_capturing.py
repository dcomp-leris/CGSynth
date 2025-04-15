'''
Copyright 2025 LERIS Lab 

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

'''
''' ************ Project Code Updates *************** 
# Project: CGReplay
# Main Module: Capturing
# sub Module: Screen + Joystick
# Date: 2025-03-20 / time: 14:26
# Author(s): Alireza Shirmarz
# Location: Leris lab
''' ************************************************* 


import subprocess
import time
import yaml

# Load configuration from YAML file
with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load settings from YAML file
network_interface = config["network_interface"]
capture_duration = config["duration"]
filename = config["filename_prefix"]
script1 = config["script_joystick"]
script2 = config["script_screen"]
waiting_time = config["starting_waiting_time"]

enable_joystick =  config["capturing_options"]["enable_joystick"]
enable_screen = config["capturing_options"]["enable_screen"]
enable_pcap= config["capturing_options"]["enable_pcap"]

try:
    print(f"Waiting {waiting_time} seconds to start logging .... ")
    time.sleep(waiting_time)

    # Start the Python scripts
    if enable_joystick == True:
        process1 = subprocess.Popen(['python3', script1, filename])
    if enable_screen == True:
        process2 = subprocess.Popen(['python3', script2, filename])
    if enable_pcap == True:
        # Start tshark command to capture network traffic
        process3 = subprocess.Popen([
            'sudo', 'tshark', '-i', network_interface,
            '-a', f'duration:{capture_duration}', '-w', f'{filename}.pcap'
        ])

    # Wait for the Python scripts to finish
    if enable_joystick == True:
        process1.wait()
    if enable_screen == True:
        process2.wait()
        # Optionally wait for tshark to finish (if needed)
    if enable_pcap == True:
        process3.wait()

    print("All processes finished successfully.")

except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")
except KeyboardInterrupt:
    print("Processes interrupted by user.")
finally:
    # Make sure to terminate tshark and other processes if necessary
    process1.terminate()
    process2.terminate()
    process3.terminate()
    print("All processes terminated.")
