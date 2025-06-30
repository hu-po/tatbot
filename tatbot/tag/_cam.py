"""Camera module for handling PoE IP cameras using ffmpeg."""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import ffmpeg
import numpy as np
import yaml
from dacite import from_dict

from tatbot.utils.log import (TIME_FORMAT, get_logger, print_config,
                              setup_log_with_config)

log = get_logger('tag.cam', '📸')

@dataclass
class CameraManagerConfig:
    """Configuration for the camera manager."""
    cameras: dict[str, CameraConfig]
    output_dir: str
    image_format: str
    image_quality: int

class CameraManager:
    """Manages multiple PoE IP cameras using ffmpeg."""
    
    def __init__(self, config_path: str):
        """Initialize camera manager with config file."""
        self.config = self._load_config(config_path)
        self.processes: Dict[str, Tuple[ffmpeg.Stream, ffmpeg.Stream]] = {}
        os.makedirs(self.config.output_dir, exist_ok=True)
        log.info(f"📸 🎬 Initialized camera manager with {len(self.config.cameras)} cameras")
        
    def _load_config(self, config_path: str) -> CameraManagerConfig:
        """Load camera configuration from yaml file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        # Replace environment variables in passwords
        for cam_config in config_dict['cameras'].values():
            if cam_config['password'].startswith('${') and cam_config['password'].endswith('}'):
                env_var = cam_config['password'][2:-1]
                cam_config['password'] = os.environ.get(env_var, '')
        log.debug(f"📸 📝 Loaded camera configuration from {config_path}")
        return from_dict(data_class=CameraManagerConfig, data=config_dict)

    def start_camera(self, camera_id: str) -> None:
        """Start capturing from a specific camera."""
        if camera_id not in self.config.cameras:
            raise ValueError(f"📸 ❌ Camera {camera_id} not found in config")
        
        cam_config = self.config.cameras[camera_id]
        rtsp_url = f"rtsp://{cam_config.username}:{cam_config.password}@{cam_config.ip}:{cam_config.rtsp_port}{cam_config.stream_path}"
        
        try:
            # Setup ffmpeg stream
            stream = (
                ffmpeg
                .input(
                    rtsp_url,
                    rtsp_transport='tcp',  # More reliable than UDP
                    r=cam_config.fps
                )
                .output(
                    'pipe:',
                    format='rawvideo',
                    pix_fmt='rgb24',
                    s=f'{cam_config.resolution[0]}x{cam_config.resolution[1]}'
                )
                .overwrite_output()
            )
            
            # Start the ffmpeg process
            process = stream.run_async(pipe_stdout=True, pipe_stderr=True)
            self.processes[camera_id] = (stream, process)
            log.info(f"📸 ✅ Started camera {camera_id}")
            
        except ffmpeg.Error as e:
            log.error(f"📸 ❌ Failed to start camera {camera_id}: {str(e)}")
            raise

    def stop_camera(self, camera_id: str) -> None:
        """Stop capturing from a specific camera."""
        if camera_id in self.processes:
            _, process = self.processes[camera_id]
            process.kill()
            del self.processes[camera_id]
            log.info(f"📸 🛑 Stopped camera {camera_id}")

    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get the latest frame from a camera."""
        if camera_id not in self.processes:
            log.warning(f"📸 ⚠️ Camera {camera_id} is not started")
            return None
        
        try:
            cam_config = self.config.cameras[camera_id]
            _, process = self.processes[camera_id]
            
            # Read raw bytes
            frame_size = cam_config.resolution[0] * cam_config.resolution[1] * 3
            in_bytes = process.stdout.read(frame_size)
            
            if not in_bytes:
                log.warning(f"📸 ⚠️ No data received from camera {camera_id}")
                return None
                
            # Convert to numpy array
            frame = (
                np.frombuffer(in_bytes, np.uint8)
                .reshape([cam_config.resolution[1], cam_config.resolution[0], 3])
            )
            
            log.debug(f"📸 🎥 Got frame from camera {camera_id}")
            return frame
            
        except Exception as e:
            log.error(f"📸 ❌ Error getting frame from camera {camera_id}: {str(e)}")
            return None

    def save_frame(self, camera_id: str, frame: np.ndarray) -> Optional[str]:
        """Save a frame to disk."""
        try:
            timestamp = datetime.now().strftime(TIME_FORMAT)
            filename = f"{camera_id}_{timestamp}.{self.config.image_format}"
            filepath = os.path.join(self.config.output_dir, filename)
            
            # Save using OpenCV (supports more formats than PIL)
            cv2.imwrite(filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            log.debug(f"📸 💾 Saved frame from {camera_id} to {filepath}")
            return filepath
            
        except Exception as e:
            log.error(f"📸 ❌ Error saving frame from camera {camera_id}: {str(e)}")
            return None

    def start_all(self) -> None:
        """Start all cameras."""
        log.info(f"📸 🎬 Starting all cameras...")
        for camera_id in self.config.cameras:
            self.start_camera(camera_id)

    def stop_all(self) -> None:
        """Stop all cameras."""
        log.info(f"📸 🛑 Stopping all cameras...")
        for camera_id in list(self.processes.keys()):
            self.stop_camera(camera_id)

    def __enter__(self):
        """Context manager entry."""
        self.start_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_all() 