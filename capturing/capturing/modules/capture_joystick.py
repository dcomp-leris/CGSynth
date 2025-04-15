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
''' ******************* Code Updates *******************
Project: CGReplay
Main Module: Capturing
sub Module: Capturing the Joystick
Date: 2025-03-10
Author: Alireza Shirmarz
Location: Lerislab
''' ****************************************************
import pygame
import time
import json
import yaml
from datetime import datetime

# Load configuration from YAML file
with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

LOG_FILE = f"{config['log_file_prefix']}.txt"
CAPTURE_DURATION = config["duration"]
TICKS_PER_SECOND = config["ticks_per_second"]

def get_joystick_data():
    """Retrieve current joystick axes and buttons data."""
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No joystick detected.")

    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]    
    axes.append(joystick.get_hat(0)[0])
    axes.append(joystick.get_hat(0)[1])
    
    buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]

    return {
        "axes": {i: axes[i] for i in range(len(axes))},
        "buttons": {i: buttons[i] for i in range(len(buttons))}
    }

def log_joystick_data():
    """Log joystick data based on sampling rate and on changes."""
    with open(LOG_FILE, "a") as log_file:
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  
        log_file.write(f"Logging started at: {start_time}\n")

        end_time = time.time() + CAPTURE_DURATION
        previous_data = None
        tick_count = 0
        tick_interval = 1.0 / TICKS_PER_SECOND
        next_tick_time = time.time() + tick_interval

        while time.time() < end_time:
            current_time = time.time()

            current_data = get_joystick_data()

            # Check for changes between ticks
            if current_data != previous_data:
                print(current_data)
                log_entry = create_log_entry(tick_count, current_data)
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()  
                previous_data = current_data  

            # Log data at the next tick
            if current_time >= next_tick_time:
                tick_count += 1
                next_tick_time += tick_interval
                log_entry = create_log_entry(tick_count, current_data)
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()  

            time.sleep(0.01)

        end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_file.write(f"Logging ended at: {end_time_str}\n")

def create_log_entry(tick_count, data):
    """Helper function to create a log entry with milliseconds."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  
    return {
        "tick": tick_count,
        "timestamp": timestamp,
        "data": data
    }

if __name__ == "__main__":
    print("Joystick logger starts...\n")
    log_joystick_data()
