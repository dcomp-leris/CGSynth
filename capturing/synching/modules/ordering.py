'''
Project: CGReplay
Main Module: Sync
sub Module: ordering
Date: 2025-03-10
Author(s): Alireza Shirmarz
Location: Lerislab
'''

import pandas as pd
import hashlib
import ast
import yaml

# Load configuration from YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

input_log = config["mix_file"]
output_log = config["output_log"]

# Function to hash the command
def hash_string(input_string, output_size):
    """Hashes a string using SHAKE and returns the hex digest with the given output size in bytes."""
    shake = hashlib.shake_128()
    shake.update(input_string.encode('utf-8'))
    return shake.hexdigest(output_size)

# Read the CSV data
df = pd.read_csv(input_log)

# Initialize an empty list to hold the extracted frames
frames = []

# Open the log file to write
with open(output_log, 'w') as log_file:
    log_file.write('ID,command,encrypted_cmd\n')  # Write the header
    
    # Iterate through the dataframe
    for index, row in df.iterrows():
        if row['Type'] == 'Joystick':  # If the row is of type Joystick
            if index > 0:  # Ensure there's a previous row
                prev_row = df.iloc[index - 1]
                if prev_row['Type'] == 'Frame':
                    # Extract Frame ID and add to frames list
                    frame_id = int(prev_row['Data'].split(',')[0].split(':')[1].strip())
                    frames.append(frame_id)
                    
                    # Collect all joystick rows following this Frame
                    next_index = index
                    while next_index < len(df) and df.iloc[next_index]['Type'] == 'Joystick':
                        joystick_row = df.iloc[next_index]
                        joystick_data = ast.literal_eval(joystick_row['Data'])['data']
                        
                        # Prepare the log entry
                        command = str(joystick_data)
                        encrypted_cmd = hash_string(command, 45)
                        
                        # Write to the log
                        log_file.write(f"{frame_id},{command},{encrypted_cmd}\n")
                        
                        next_index += 1

# Print the extracted frames list
print("Frames list:", frames)
