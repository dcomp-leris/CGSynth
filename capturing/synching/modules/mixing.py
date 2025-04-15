'''
Project: CGReplay
Main Module: Sync
sub Module: Mixing 
Date: 2025-03-10
Author(s): Alireza Shirmarz
Location: Lerislab
'''
import csv
import yaml
from datetime import datetime

# Load configuration from YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

frames_log = config["output_file"]
joystick_log = config["joystick_log"]
mix_file = config["mix_file"]

def parse_timestamp(timestamp):
    """Convert timestamp string to a datetime object."""
    return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")

def read_frames_log(frames_log):
    """Read the frame log and return a list of dictionaries."""
    frames = []
    with open(frames_log, 'r') as f:
        for line in f:
            if "Frame" in line:
                try:
                    parts = line.split(',')
                    frame_name = parts[0].split(': ')[1].strip()  # Get the frame name
                    frame_id = parts[1].split(': ')[2].strip()    # Get the correct frame ID
                    timestamp = parts[2].split(': ')[1].strip()   # Get the timestamp

                    frames.append({
                        'type': 'Frame',
                        'timestamp': parse_timestamp(timestamp),
                        'frame_id': frame_id,
                        'frame_name': frame_name
                    })
                except (IndexError, ValueError) as e:
                    print(f"Error processing line: {line}. Error: {e}")
    return frames

def read_joystick_log(joystick_log):
    """Read the joystick log and return a list of dictionaries."""
    joystick_data = []
    with open(joystick_log, 'r') as f:
        for line in f:
            if "tick" in line:
                entry = eval(line.strip())  # Convert the log entry string into a Python dictionary
                timestamp = entry['timestamp']
                joystick_data.append({
                    'type': 'Joystick',
                    'timestamp': parse_timestamp(timestamp),
                    'data': entry
                })
    return joystick_data

def write_ordered_log(ordered_data, mix_file):
    """Write the ordered data into the output log file."""
    with open(mix_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Type', 'Timestamp', 'Data'])
        for idx, item in enumerate(ordered_data, start=1):
            if item['type'] == 'Frame':
                row = [idx, 'Frame', item['timestamp'], f"Frame ID: {item['frame_id']}, Frame Name: {item['frame_name']}"]
            else:
                row = [idx, 'Joystick', item['timestamp'], item['data']]
            writer.writerow(row)

def merge_and_order_logs(frames_log, joystick_log, mix_file):
    """Merge the frame and joystick logs and order them by timestamp."""
    frames = read_frames_log(frames_log)
    joysticks = read_joystick_log(joystick_log)
    
    # Combine both lists
    combined_logs = frames + joysticks
    
    # Sort by timestamp
    ordered_logs = sorted(combined_logs, key=lambda x: x['timestamp'])
    
    # Write to the output log
    write_ordered_log(ordered_logs, mix_file)

# Call the function to merge and order the logs
merge_and_order_logs(frames_log, joystick_log, mix_file)

print(f"Logs merged and ordered. Output written to {mix_file}.")


