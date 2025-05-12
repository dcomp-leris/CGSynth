import cv2
import numpy as np
import argparse
import os
import random
from tqdm import tqdm
from scipy.ndimage import zoom

class CloudGamingNoise:
    def __init__(self, seed=None):
        """Initialize the noise generator with optional seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Define effect categories
        self.effect_categories = {
            'network': [
                'compression_quality',  # Compression artifacts
                'macroblock_loss',      # Packet loss
                'resolution_scale',     # Bandwidth adaptation
                'frame_freeze'          # Buffer underruns
            ],
            'rendering': [
                'motion_blur',          # Not typically network-related
                'color_banding',        # Can be codec-related but also GPU
                'quantization_noise'    # More codec-specific than network
            ]
        }
        
    def add_compression_artifacts(self, frame, quality=50):
        """Add JPEG compression artifacts to simulate low bitrate encoding."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', frame, encode_param)
        return cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    
    def add_macroblock_loss(self, frame, block_size=16, num_blocks=10):
        """Simulate packet loss by corrupting random macroblocks."""
        result = frame.copy()
        height, width = frame.shape[:2]
        
        for _ in range(num_blocks):
            # Choose random position for macroblock
            x = random.randint(0, width - block_size)
            y = random.randint(0, height - block_size)
            
            # Replace with either black, previous content with noise, or shifted content
            effect = random.choice(['black', 'noise', 'shift'])
            
            if effect == 'black':
                result[y:y+block_size, x:x+block_size] = 0
            elif effect == 'noise':
                noise = np.random.randint(0, 255, (block_size, block_size, 3), dtype=np.uint8)
                result[y:y+block_size, x:x+block_size] = noise
            else:  # shift
                shift_x = random.randint(-5, 5)
                shift_y = random.randint(-5, 5)
                
                src_x = max(0, x + shift_x)
                src_y = max(0, y + shift_y)
                src_x2 = min(width, src_x + block_size)
                src_y2 = min(height, src_y + block_size)
                
                if src_x2 > src_x and src_y2 > src_y:
                    block = frame[src_y:src_y2, src_x:src_x2].copy()
                    h, w = block.shape[:2]
                    result[y:y+h, x:x+w] = block
        
        return result
    
    def add_motion_blur(self, frame, size=15, angle=45):
        """Add directional motion blur."""
        # Create motion blur kernel
        k = np.zeros((size, size), dtype=np.float32)
        k[(size-1)//2, :] = np.ones(size, dtype=np.float32)
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((size/2-0.5, size/2-0.5), angle, 1.0), (size, size))
        k = k / np.sum(k)
        
        # Apply the kernel
        return cv2.filter2D(frame, -1, k)
    
    def add_color_banding(self, frame, levels=32):
        """Reduce color depth to create banding artifacts."""
        factor = 255.0 / levels
        return np.uint8(np.floor(frame / factor) * factor)
    
    def simulate_resolution_change(self, frame, scale_factor=0.5):
        """Simulate sudden resolution drop and upscaling."""
        h, w = frame.shape[:2]
        
        # Downscale
        small = cv2.resize(frame, (int(w*scale_factor), int(h*scale_factor)), 
                           interpolation=cv2.INTER_LINEAR)
        
        # Upscale back to original size
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    def add_quantization_noise(self, frame, strength=10):
        """Add quantization noise that mimics codec artifacts."""
        noise = np.random.normal(0, strength, frame.shape).astype(np.int16)
        
        # Add noise and clip to valid range
        noisy_frame = frame.astype(np.int16) + noise
        return np.clip(noisy_frame, 0, 255).astype(np.uint8)
    
    def simulate_frame_freeze(self, frames, freeze_probability=0.05, freeze_duration=3):
        """Simulate frame freezing by duplicating frames."""
        result = []
        i = 0
        
        while i < len(frames):
            if random.random() < freeze_probability:
                # Freeze frame for duration
                freeze_frame = frames[i]
                for _ in range(freeze_duration):
                    result.append(freeze_frame.copy())
                i += 1
            else:
                result.append(frames[i])
                i += 1
                
        return result
    
    def create_network_degradation_profile(self, num_frames, severity=0.5, effect_types=None):
        """
        Create a time-varying profile of network conditions.
        Returns a list of dictionaries with degradation parameters for each frame.
        
        Args:
            num_frames: Number of frames to generate profile for
            severity: Overall severity of artifacts (0.0-1.0)
            effect_types: List of effect types to include ('network', 'rendering', or 'all')
        """
        # Determine which effects to include
        included_effects = []
        if effect_types is None or 'all' in effect_types:
            included_effects = self.effect_categories['network'] + self.effect_categories['rendering']
        else:
            for category in effect_types:
                if category in self.effect_categories:
                    included_effects.extend(self.effect_categories[category])
        
        # Base parameters (all set to no effect)
        base_params = {
            'compression_quality': 90,  # Higher is better
            'macroblock_loss': 0,       # Number of affected blocks
            'motion_blur': 0,           # Blur kernel size
            'color_banding': 0,         # 0 means no banding
            'resolution_scale': 1.0,    # 1.0 means no change
            'quantization_noise': 0,    # Noise strength
            'frame_freeze': False       # Flag for frame freezing
        }
        
        # Create profiles
        profiles = []
        
        # Generate some random network events
        num_events = int(num_frames * 0.2 * severity)  # More events with higher severity
        event_frames = sorted(random.sample(range(num_frames), min(num_events, num_frames)))
        
        # Generate event durations (in frames)
        event_durations = [random.randint(1, max(2, int(30 * severity))) for _ in range(len(event_frames))]
        
        # Create clean profile first
        for _ in range(num_frames):
            profiles.append(base_params.copy())
        
        # Apply network events
        for event_start, duration in zip(event_frames, event_durations):
            event_end = min(event_start + duration, num_frames)
            
            # Only select from included effects
            available_event_types = []
            if 'compression_quality' in included_effects:
                available_event_types.append('compression')
            if 'macroblock_loss' in included_effects:
                available_event_types.append('packet_loss')
            if 'motion_blur' in included_effects:
                available_event_types.append('blur')
            if 'color_banding' in included_effects:
                available_event_types.append('banding')
            if 'resolution_scale' in included_effects:
                available_event_types.append('resolution')
            if 'quantization_noise' in included_effects:
                available_event_types.append('quantization')
            
            # Add combined type if we have multiple effects
            if len(available_event_types) > 1:
                available_event_types.append('combined')
            
            # If no effects included, skip this event
            if not available_event_types:
                continue
                
            event_type = random.choice(available_event_types)
            
            # Apply the chosen degradation
            for i in range(event_start, event_end):
                # Ramp effect intensity based on position in the event
                position_factor = min(1.0, (i - event_start + 1) / 5)  # Ramp up over 5 frames
                decay_factor = min(1.0, (event_end - i) / 5)  # Decay over 5 frames
                intensity = position_factor * decay_factor * severity
                
                if (event_type == 'compression' or event_type == 'combined') and 'compression_quality' in included_effects:
                    # Lower quality means more compression artifacts
                    profiles[i]['compression_quality'] = max(5, int(90 - 85 * intensity))
                
                if (event_type == 'packet_loss' or event_type == 'combined') and 'macroblock_loss' in included_effects:
                    # More lost macroblocks with higher intensity
                    profiles[i]['macroblock_loss'] = int(20 * intensity)
                
                if (event_type == 'blur' or event_type == 'combined') and 'motion_blur' in included_effects:
                    profiles[i]['motion_blur'] = int(25 * intensity) * 2 + 1  # Ensure odd kernel size
                
                if (event_type == 'banding' or event_type == 'combined') and 'color_banding' in included_effects:
                    # Fewer color levels means more banding
                    color_levels = max(4, int(256 - 232 * intensity))
                    profiles[i]['color_banding'] = color_levels if color_levels < 256 else 0
                
                if (event_type == 'resolution' or event_type == 'combined') and 'resolution_scale' in included_effects:
                    # Lower scale factor means more resolution loss
                    profiles[i]['resolution_scale'] = max(0.1, 1.0 - 0.9 * intensity)
                
                if (event_type == 'quantization' or event_type == 'combined') and 'quantization_noise' in included_effects:
                    profiles[i]['quantization_noise'] = int(30 * intensity)
                
                # Mark frames for potential freezing (actual freezing happens later)
                if 'frame_freeze' in included_effects and random.random() < 0.01 * severity:
                    profiles[i]['frame_freeze'] = True
        
        return profiles
    
    def process_frame(self, frame, params):
        """Apply degradation effects to a frame based on parameters."""
        result = frame.copy()
        
        # Apply effects in a sensible order
        if params['resolution_scale'] < 1.0:
            result = self.simulate_resolution_change(result, params['resolution_scale'])
            
        if params['compression_quality'] < 90:
            result = self.add_compression_artifacts(result, params['compression_quality'])
            
        if params['macroblock_loss'] > 0:
            result = self.add_macroblock_loss(result, num_blocks=params['macroblock_loss'])
            
        if params['color_banding'] > 0:
            result = self.add_color_banding(result, params['color_banding'])
            
        if params['motion_blur'] > 0:
            result = self.add_motion_blur(result, size=params['motion_blur'])
            
        if params['quantization_noise'] > 0:
            result = self.add_quantization_noise(result, params['quantization_noise'])
            
        return result


def main():
    parser = argparse.ArgumentParser(description="Add realistic network artifacts to video frames")
    parser.add_argument("--input_dir", required=True, help="Directory containing input frames (as images)")
    parser.add_argument("--output_dir", required=True, help="Directory to save processed frames")
    parser.add_argument("--severity", type=float, default=0.5, help="Overall severity of artifacts (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--frame_pattern", default="frame_%04d.png", help="Pattern for frame filenames")
    parser.add_argument("--effect_types", choices=['network', 'rendering', 'all'], default='all', 
                       nargs='+', help="Type of effects to apply")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of files
    input_files = sorted([f for f in os.listdir(args.input_dir) 
                         if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if not input_files:
        print(f"No image files found in {args.input_dir}")
        return
    
    # Initialize noise generator
    noise_gen = CloudGamingNoise(seed=args.seed)
    
    # Create degradation profile
    print("Creating network degradation profile...")
    profile = noise_gen.create_network_degradation_profile(
        len(input_files), 
        args.severity, 
        args.effect_types
    )
    
    # Process frames
    print(f"Processing {len(input_files)} frames...")
    processed_frames = []
    
    for i, filename in enumerate(tqdm(input_files)):
        # Read frame
        frame_path = os.path.join(args.input_dir, filename)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Error reading {frame_path}")
            continue
        
        # Apply degradation
        degraded_frame = noise_gen.process_frame(frame, profile[i])
        processed_frames.append(degraded_frame)
        
        # Save processed frame
        output_path = os.path.join(args.output_dir, filename)
        cv2.imwrite(output_path, degraded_frame)
    
    # Simulate frame freezes if included in effects
    should_freeze = args.severity > 0.2 and ('all' in args.effect_types or 'network' in args.effect_types)
    
    if should_freeze:
        print("Simulating frame freezes...")
        frozen_frames = noise_gen.simulate_frame_freeze(processed_frames, 
                                                       freeze_probability=0.02 * args.severity,
                                                       freeze_duration=int(5 * args.severity))
        
        # Save the frames with freezes
        for i, frame in enumerate(tqdm(frozen_frames)):
            output_path = os.path.join(args.output_dir, args.frame_pattern % i)
            cv2.imwrite(output_path, frame)
    else:
        print("Frame freezing skipped (not included in selected effects)")
    
    print(f"Processing complete. {len(input_files)} frames processed.")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()