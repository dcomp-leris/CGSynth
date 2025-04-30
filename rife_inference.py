import socket
import os
import sys
import cv2
import numpy as np
from pathlib import Path
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_rife_repo_path():
    """Get the path to the ECCV2022-RIFE repository."""
    return os.path.expanduser('~/git/ECCV2022-RIFE')

def setup_rife_environment():
    """Add RIFE repository to Python path and import necessary modules."""
    rife_repo_path = get_rife_repo_path()
    if rife_repo_path not in sys.path:
        sys.path.append(rife_repo_path)
    
    try:
        from rife_inference import inference
        # Initialize model by calling inference once
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        inference(dummy_frame, dummy_frame)
        return inference
    except ImportError as e:
        logger.error(f"Error importing RIFE modules: {e}")
        sys.exit(1)

def process_request(frame1_path, frame3_path, output_path):
    """Process interpolation request using RIFE."""
    try:
        # Read input frames
        frame1 = cv2.imread(frame1_path)
        frame3 = cv2.imread(frame3_path)
        
        if frame1 is None or frame3 is None:
            return "ERROR: Failed to read input frames"
        
        # Get RIFE inference function
        inference = setup_rife_environment()
        
        # Run interpolation
        interpolated_frame = inference(frame1, frame3)
        
        # Save result
        cv2.imwrite(output_path, interpolated_frame)
        return "OK"
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return f"ERROR: {str(e)}"

def main():
    # Initialize RIFE environment first
    logger.info("Initializing RIFE environment...")
    setup_rife_environment()
    logger.info("RIFE environment initialized successfully")
    
    # Create socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        # Bind to localhost on port 50051
        server_socket.bind(('localhost', 50051))
        server_socket.listen(1)
        
        logger.info("RIFE server started. Waiting for connections on localhost:50051...")
        
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
                    logger.error(f"Error handling client: {e}")
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