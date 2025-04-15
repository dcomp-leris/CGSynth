import subprocess
import yaml

# Load configuration from YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Paths to the three scripts
script1 = config["script1"]
script2 = config["script2"]
script3 = config["script3"]

# Function to run a script and wait for its completion
def run_script(script_name):
    print(f"Running {script_name}...")
    process = subprocess.run(["python3", script_name])
    
    # Check if script executed successfully
    if process.returncode == 0:
        print(f"{script_name} completed successfully.\n")
    else:
        print(f"Error: {script_name} encountered an issue!\n")
        exit(1)  # Exit if a script fails

# Run the scripts one by one
run_script(script1)
run_script(script2)
run_script(script3)

print("Synchronization was done! 🎉")
