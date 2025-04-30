import socket
import os
import sys
import cv2
import numpy as np
from pathlib import Path
import torch
import logging
from torch.nn import functional as F
import warnings
import time
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path to the RIFE repository
RIFE_REPO_PATH = os.path.dirname(os.path.abspath(__file__))

class RIFEInference:
    def __init__(self, model_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        
        # Use the train_log directory in the RIFE repository
        self.model_dir = os.path.join(RIFE_REPO_PATH, 'train_log')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Check for the main model file
        model_file = 'flownet.pkl'
        if not os.path.exists(os.path.join(self.model_dir, model_file)):
            logger.error(f"Missing model file: {model_file}")
            logger.error(f"Please download the model file from https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HpcMHqEMo")
            logger.error(f"and place it in: {self.model_dir}")
            raise FileNotFoundError(f"Missing model file in {self.model_dir}")
        
        # Try to load different model versions
        try:
            try:
                try:
                    from model.RIFE_HDv2 import Model
                    self.model = Model()
                    self.model.load_model(self.model_dir, -1)
                    logger.info("Loaded v2.x HD model.")
                except:
                    from train_log.RIFE_HDv3 import Model
                    self.model = Model()
                    self.model.load_model(self.model_dir, -1)
                    logger.info("Loaded v3.x HD model.")
            except:
                from model.RIFE_HD import Model
                self.model = Model()
                self.model.load_model(self.model_dir, -1)
                logger.info("Loaded v1.x HD model")
        except:
            from model.RIFE import Model
            self.model = Model()
            self.model.load_model(self.model_dir, -1)
            logger.info("Loaded ArXiv-RIFE model")
        
        self.model.eval()
        self.model.device()

    def preprocess_frame(self, frame):
        """Convert OpenCV frame to tensor format."""
        frame = (torch.tensor(frame.transpose(2, 0, 1)).to(self.device) / 255.).unsqueeze(0)
        n, c, h, w = frame.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        return F.pad(frame, padding), h, w

    def postprocess_frame(self, tensor, h, w):
        """Convert tensor back to OpenCV frame format."""
        return (tensor[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

    def interpolate(self, frame1, frame2):
        """Interpolate between two frames."""
        # Preprocess frames
        frame1_tensor, h, w = self.preprocess_frame(frame1)
        frame2_tensor, _, _ = self.preprocess_frame(frame2)
        
        # Run inference
        with torch.no_grad():
            interpolated_tensor = self.model.inference(frame1_tensor, frame2_tensor)
        
        # Postprocess result
        return self.postprocess_frame(interpolated_tensor, h, w)

def process_request(frame1_path, frame3_path, output_path):
    """Process interpolation request using RIFE."""
    try:
        # Read input frames
        frame1 = cv2.imread(frame1_path)
        frame3 = cv2.imread(frame3_path)
        
        if frame1 is None or frame3 is None:
            return "ERROR: Failed to read input frames"
        
        # Run interpolation
        rife = RIFEInference()
        interpolated_frame = rife.interpolate(frame1, frame3)
        
        # Save result
        cv2.imwrite(output_path, interpolated_frame)
        return "OK"
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return f"ERROR: {str(e)}"

def check_port_in_use(port, host='localhost'):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except socket.error:
            return True

def find_available_port(start_port=50051, max_attempts=10):
    """Find an available port starting from start_port."""
    for port_offset in range(max_attempts):
        port = start_port + port_offset
        if not check_port_in_use(port):
            return port
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

def kill_previous_instance(port):
    """Try to identify and kill a process running on the specified port (platform-specific)."""
    try:
        if sys.platform == 'win32':
            # For Windows
            os.system(f'for /f "tokens=5" %p in (\'netstat -ano ^| findstr :{port}\') do taskkill /F /PID %p')
        else:
            # For Linux/Mac
            os.system(f"lsof -ti:{port} | xargs kill -9")
        logger.info(f"Attempted to kill previous process on port {port}")
        time.sleep(1)  # Give the OS time to free up the port
    except Exception as e:
        logger.warning(f"Could not kill previous process: {e}")

def main():
    PORT = 50051
    HOST = 'localhost'
    MAX_RETRY = 3
    
    # Create socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Try to bind to the port, with retries
    retry_count = 0
    while retry_count < MAX_RETRY:
        try:
            server_socket.bind((HOST, PORT))
            break
        except socket.error as e:
            if e.errno == 98:  # Address already in use
                retry_count += 1
                logger.warning(f"Port {PORT} is already in use (attempt {retry_count}/{MAX_RETRY})")
                
                if retry_count == 1:
                    # First try: attempt to kill the previous instance
                    logger.info("Attempting to kill the previous server instance...")
                    kill_previous_instance(PORT)
                elif retry_count == 2:
                    # Second try: wait a bit longer
                    logger.info("Waiting for port to become available...")
                    time.sleep(5)
                else:
                    # Last try: find a different port
                    try:
                        PORT = find_available_port(PORT + 1)
                        logger.info(f"Using alternative port: {PORT}")
                    except RuntimeError as e:
                        logger.error(str(e))
                        logger.error("Could not start server. Please ensure no other instances are running.")
                        return
            else:
                logger.error(f"Socket error: {e}")
                return
    
    try:
        server_socket.listen(1)
        logger.info(f"RIFE server started. Waiting for connections on {HOST}:{PORT}...")
        
        while True:
            try:
                # Accept connection
                client_socket, address = server_socket.accept()
                logger.info(f"Connection from {address}")
                
                try:
                    # Receive request
                    data = client_socket.recv(1024).decode()
                    if not data:
                        continue
                    
                    # Parse request
                    if data.strip() == "EXIT":
                        logger.info("Received exit command")
                        break
                    
                    frame1_path, frame3_path, output_path = data.strip().split('|')
                    logger.info(f"Processing request: {frame1_path} -> {output_path}")
                    
                    # Process request
                    response = process_request(frame1_path, frame3_path, output_path)
                    
                    # Send response
                    client_socket.sendall(response.encode())
                    
                except Exception as e:
                    error_msg = f"ERROR: {str(e)}"
                    logger.error(f"Error handling client: {e}")
                    try:
                        client_socket.sendall(error_msg.encode())
                    except:
                        pass
                finally:
                    client_socket.close()
                    
            except KeyboardInterrupt:
                logger.info("\nShutting down server...")
                break
                
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        server_socket.close()
        logger.info("Server socket closed")

if __name__ == "__main__":
    main()