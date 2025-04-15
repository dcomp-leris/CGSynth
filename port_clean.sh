#!/bin/bash

# List of UDP ports to check and kill processes
ports=("5000" "5001"  "5002" "5003" "5004" "5005" "5006" "5007" "5008" "5009" "5010")  # Add more ports as needed
#ports=({5000..5020})

# Path to the YAML file
#yaml_file="./config/config.yaml"

# Extract UDP ports from the YAML file
#ports=($(yq eval '.server_port, .server_command, .player_streaming_port, .palyer_command_port' "$yaml_file"))  # Extract multiple values



for port in "${ports[@]}"; do
  # Find the process using the current UDP port
  pid=$(sudo lsof -t -i UDP:$port)

  # Check if a process was found for the port
  if [ -z "$pid" ]; then
    echo "No process found using UDP port $port."
  else
    # Kill the process by PID
    sudo kill -9 $pid
    echo "Process $pid using UDP port $port has been killed."
  fi
done

