import socket
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import logging
import time
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FILMInference:
    def __init__(self):
        self.device = "cuda" if tf.config.list_physical_devices('GPU') else "cpu"
        
        # Configure TensorFlow for GPU if available
        if self.device == "cuda":
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    logger.error(f"Memory growth setting error: {e}")
            
            # Optimize thread settings
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
        
        # Load the FILM model
        logger.info("Loading FILM model from TensorFlow Hub...")
        try:
            self.model = hub.load("https://tfhub.dev/google/film/1")
            logger.info("FILM model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading FILM model: {e}")
            raise

    def preprocess_frame(self, frame):
        """Convert OpenCV frame to format expected by FILM model."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to float and normalize to [0, 1]
        frame_float = frame_rgb.astype(np.float32) / 255.0
        return frame_float

    def postprocess_frame(self, frame_tensor):
        """Convert model output back to OpenCV format."""
        # Convert from [0, 1] to [0, 255] and back to uint8
        frame_uint8 = (frame_tensor.numpy() * 255.0).astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
        return frame_bgr

    def interpolate(self, frame1, frame2):
        """Interpolate between two frames using FILM."""
        # Preprocess frames
        frame1_processed = self.preprocess_frame(frame1)
        frame2_processed = self.preprocess_frame(frame2)
        
        # Create tensors for the model
        frame1_tensor = tf.convert_to_tensor(frame1_processed)
        frame2_tensor = tf.convert_to_tensor(frame2_processed)
        
        # Add batch dimension
        frame1_tensor = tf.expand_dims(frame1_tensor, 0)
        frame2_tensor = tf.expand_dims(frame2_tensor, 0)
        
        # Generate intermediate frame (time = 0.5 for middle frame)
        time_step = tf.constant([[0.5]], dtype=tf.float32)  # Shape needs to be [batch_size, 1]
        
        # Create the input dictionary as expected by the model
        inputs = {
            'x0': frame1_tensor,  # First frame
            'x1': frame2_tensor,  # Second frame
            'time': time_step     # Time step for interpolation
        }
        
        # Run the model
        output_dict = self.model(inputs, training=False)
        
        # Extract the interpolated frame
        if isinstance(output_dict, dict) and 'image' in output_dict:
            interpolated_tensor = output_dict['image']
            # Remove batch dimension if present
            if len(interpolated_tensor.shape) == 4:
                interpolated_tensor = interpolated_tensor[0]
            
            # Convert back to OpenCV format
            return self.postprocess_frame(interpolated_tensor)
        else:
            raise ValueError("Unexpected model output format")

def process_request(frame1_path, frame2_path, output_path):
    """Process interpolation request using FILM."""
    try:
        # Read input frames
        logger.info(f"Attempting to read frame1 from: {frame1_path}")
        frame1 = cv2.imread(frame1_path)
        if frame1 is None:
            logger.error(f"Failed to read frame1 from {frame1_path}")
            return "ERROR: Failed to read frame1"
            
        logger.info(f"Attempting to read frame2 from: {frame2_path}")
        frame2 = cv2.imread(frame2_path)
        if frame2 is None:
            logger.error(f"Failed to read frame2 from {frame2_path}")
            return "ERROR: Failed to read frame2"
        
        logger.info(f"Successfully read both frames. Frame1 shape: {frame1.shape}, Frame2 shape: {frame2.shape}")
        
        # Run interpolation
        film = FILMInference()
        interpolated_frame = film.interpolate(frame1, frame2)
        
        # Save result
        logger.info(f"Saving interpolated frame to: {output_path}")
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

def find_available_port(start_port=50052, max_attempts=10):
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
    PORT = 50052  # Different port from RIFE server
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
        logger.info(f"FILM server started. Waiting for connections on {HOST}:{PORT}...")
        
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
                    
                    frame1_path, frame2_path, output_path = data.strip().split('|')
                    logger.info(f"Processing request: {frame1_path} -> {output_path}")
                    
                    # Process request
                    response = process_request(frame1_path, frame2_path, output_path)
                    
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