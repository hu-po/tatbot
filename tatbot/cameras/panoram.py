# $TATBOT_ROOT/tatbot/cameras/panoram.py

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
import time
from typing import Dict, Optional, Literal

import cv2
import requests
from requests.auth import HTTPDigestAuth
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

@dataclass
class Camera:
    name: str
    device_name: str
    ip: str
    MAC: str
    username: str
    password: str = None

@dataclass
class PanoramConfig:
    # File formats and constants
    image_format: str = "jpg"
    video_format: str = "avi"
    video_codec: str = "MJPG"
    video_fps: float = 30.0
    video_duration: int = 5
    filename_timestamp_format: str = "%Yy%mm%dd%Hh%Mm%Ss"
    rtsp_url_suffix: str = ":554/cam/realmonitor?channel=1&subtype=0"
    
    # Paths and directories
    root_dir: str = os.environ.get("TATBOT_ROOT")
    config_path: str = f"{root_dir}/config/cameras.yaml"
    output_dir: str = f"{root_dir}/output/panoram"
    
    # Camera configuration
    cameras: Dict[str, Camera] = None
    
    def __post_init__(self):
        if not self.root_dir:
            raise ValueError("TATBOT_ROOT environment variable must be set.")
        os.makedirs(self.output_dir, exist_ok=True)
        self.cameras = self._load_cameras()
    
    def _load_cameras(self) -> Dict[str, Camera]:
        """Loads camera configurations from YAML file."""
        assert os.path.exists(self.config_path), f"Camera config not found: {self.config_path}"
        with open(self.config_path) as f:
            data = yaml.safe_load(f)
        cameras = {}
        for key, value in data.items():
            value['name'] = key
            camera = Camera(**value)
            cameras[key] = camera
            cameras[key].password = os.getenv(f"{str.upper(key)}_PASSWORD")
        return cameras

    def make_rtsp_url(self, camera: Camera) -> str:
        """Creates RTSP URL for a camera."""
        return f"rtsp://{camera.username}:{camera.password}@{camera.ip}{self.rtsp_url_suffix}"

def get_camera_time(camera: Camera) -> Optional[datetime]:
    url = f"http://{camera.ip}/cgi-bin/global.cgi?action=getCurrentTime&channel=1"
    try:
        response = requests.get(url, auth=HTTPDigestAuth(camera.username, camera.password), timeout=5)
        response.raise_for_status()
        log.debug(f"Raw response from {camera.name}: {response.text!r}")
        
        if response.text.startswith("result="):
            time_str = response.text.split("result=")[1].strip()
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        
        log.error(f"Unexpected response format from camera {camera.name}: {response.text}")
        raise ValueError(f"Invalid time response format: {response.text}")
    except Exception as e:
        log.error(f"Error getting time from camera {camera.name} ({camera.ip}): {e}")
        return None

def verify_all_cameras_same_time(
    cameras: Dict[str, Camera],
    tolerance_seconds: float = 5.0
) -> bool:
    camera_times = {}
    for cam_name, cam in cameras.items():
        cam_time = get_camera_time(cam)
        if cam_time is None:
            print(f"Unable to retrieve time for camera {cam.name}. Failing check.")
            return False
        camera_times[cam_name] = cam_time
    times = list(camera_times.values())
    earliest = min(times)
    latest = max(times)
    delta = (latest - earliest).total_seconds()
    if delta <= tolerance_seconds:
        print(f"All cameras are within {delta:.2f} seconds of each other (tolerance={tolerance_seconds}).")
        return True
    else:
        print(f"Camera times differ by {delta:.2f} seconds, exceeding tolerance of {tolerance_seconds}.")
        return False

def toggle_camera(
    camera: Camera,
    action: Literal["enable", "disable"]
) -> bool:
    url = f"http://{camera.ip}/cgi-bin/configManager.cgi?action=setConfig&VideoInOptions[0].Enable={1 if action == 'enable' else 0}"
    try:
        response = requests.get(
            url,
            auth=HTTPDigestAuth(camera.username, camera.password),
            timeout=5
        )
        response.raise_for_status()
        log.info(f"{action.title()}d camera {camera.name}")
        return True
    except Exception as e:
        log.error(f"Failed to {action} camera {camera.name}: {e}")
        return False

async def async_toggle_all_cameras(
    cameras: Dict[str, Camera],
    action: Literal["enable", "disable"]
) -> Dict[str, bool]:
    tasks = {
        camera_id: asyncio.create_task(asyncio.to_thread(toggle_camera, camera, action))
        for camera_id, camera in cameras.items()
    }
    results_list = await asyncio.gather(*tasks.values(), return_exceptions=True)
    results = {}
    for camera_id, result in zip(tasks.keys(), results_list):
        if isinstance(result, Exception):
            log.error(f"Toggle {action} failed for camera {camera_id}: {result}")
            results[camera_id] = False
        else:
            results[camera_id] = result
    return results

def capture_image(
    config: PanoramConfig,
    camera: Camera,
    file_suffix: str,
    output_dir: str,
    filename_timestamp_format: str,
    delay: float = 0
) -> Optional[str]:
    cap = cv2.VideoCapture(config.make_rtsp_url(camera))
    if not cap.isOpened():
        log.error(f"Could not open stream for {camera.name} at {config.make_rtsp_url(camera)}")
        return None

    if delay > 0:
        time.sleep(delay)
        log.debug(f"Waited {delay}s before capturing image from {camera.name}")

    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        log.error(f"Failed to capture image for {camera.name}")
        return None

    timestamp = datetime.now().strftime(filename_timestamp_format)
    filename = f"{timestamp}_{file_suffix}.{config.image_format}"
    filepath = os.path.join(output_dir, filename)
    
    cv2.imwrite(filepath, frame)
    log.info(f"Saved image from {camera.name} ({file_suffix}) to {filepath}")
    return filepath

def capture_video(
    config: PanoramConfig,
    camera: Camera,
    file_suffix: str,
    duration: int,
    output_dir: str,
    filename_timestamp_format: str,
    video_codec: str,
    video_fps: float
) -> Optional[str]:
    cap = cv2.VideoCapture(config.make_rtsp_url(camera))
    if not cap.isOpened():
        log.error(f"Could not open stream for {camera.name} at {config.make_rtsp_url(camera)}")
        return None

    ret, frame = cap.read()
    if not ret or frame is None:
        log.error(f"Failed to read initial frame for {camera.name}")
        cap.release()
        return None

    height, width = frame.shape[:2]
    timestamp = datetime.now().strftime(filename_timestamp_format)
    filename = f"{timestamp}_{file_suffix}.{config.video_format}"
    filepath = os.path.join(output_dir, filename)

    fourcc = cv2.VideoWriter_fourcc(*video_codec)
    out = cv2.VideoWriter(filepath, fourcc, video_fps, (width, height))
    if not out.isOpened():
        log.error(f"Could not open VideoWriter with codec={video_codec}")
        cap.release()
        return None

    out.write(frame)
    frame_count = 1
    start_time = datetime.now()

    while (datetime.now() - start_time).total_seconds() < duration:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(1 / video_fps)
            continue
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    if frame_count == 0:
        if os.path.exists(filepath):
            os.remove(filepath)
        log.error(f"No frames recorded for {camera.name} ({file_suffix})")
        return None

    log.info(f"Saved video from {camera.name} ({file_suffix}) to {filepath} with {frame_count} frames")
    return filepath

async def async_get_image(config: PanoramConfig, camera: Camera, file_suffix: str, delay: float = 0) -> Optional[str]:
    return await asyncio.to_thread(
        capture_image,
        config,
        camera,
        file_suffix,
        config.output_dir,
        config.filename_timestamp_format,
        delay
    )

async def async_get_video(config: PanoramConfig, camera: Camera, file_suffix: str, duration: int) -> Optional[str]:
    return await asyncio.to_thread(
        capture_video,
        config,
        camera,
        file_suffix,
        duration,
        config.output_dir,
        config.filename_timestamp_format,
        config.video_codec,
        config.video_fps
    )

async def async_capture_all_images(config: PanoramConfig, delay: float = 0) -> Dict[str, Optional[str]]:
    tasks = {
        camera_id: asyncio.create_task(async_get_image(config, camera, f"multi_{camera.name}", delay))
        for camera_id, camera in config.cameras.items()
    }
    results_list = await asyncio.gather(*tasks.values(), return_exceptions=True)
    results = {}
    for camera_id, result in zip(tasks.keys(), results_list):
        if isinstance(result, Exception):
            log.error(f"Image capture failed for camera {camera_id}: {result}")
            results[camera_id] = None
        else:
            log.info(f"Completed image capture for camera {camera_id}: {result}")
            results[camera_id] = result
    return results

async def async_capture_all_videos(config: PanoramConfig, duration: int) -> Dict[str, Optional[str]]:
    tasks = {
        camera_id: asyncio.create_task(async_get_video(config, camera, f"multi_{camera.name}", duration))
        for camera_id, camera in config.cameras.items()
    }
    results_list = await asyncio.gather(*tasks.values(), return_exceptions=True)
    results = {}
    for camera_id, result in zip(tasks.keys(), results_list):
        if isinstance(result, Exception):
            log.error(f"Video capture failed for camera {camera_id}: {result}")
            results[camera_id] = None
        else:
            log.info(f"Completed video capture for camera {camera_id}: {result}")
            results[camera_id] = result
    return results

async def async_test(config: PanoramConfig) -> None:
    log.info("Running tests...")
    log.setLevel(logging.DEBUG)
    results: Dict[str, Dict[str, bool]] = {}
    
    log.info("\nVerifying camera times...")
    time_sync_status = verify_all_cameras_same_time(config.cameras)
    
    for camera in config.cameras.values():
        results[camera.name] = {
            "single_image": False,
            "single_video": False,
            "disable": False,
            "enable": False
        }
        try:
            image_path = await async_get_image(config, camera, "test_single")
            results[camera.name]["single_image"] = image_path is not None
        except Exception as e:
            log.error(f"Single image capture error for {camera.name}: {e}")
        
        try:
            video_path = await async_get_video(config, camera, "test_single", duration=1)
            results[camera.name]["single_video"] = video_path is not None
        except Exception as e:
            log.error(f"Single video capture error for {camera.name}: {e}")

        try:
            disable_success = await asyncio.to_thread(toggle_camera, camera, "disable")
            results[camera.name]["disable"] = disable_success
            if disable_success:
                await asyncio.sleep(2)  # Wait for disable to take effect
                enable_success = await asyncio.to_thread(toggle_camera, camera, "enable")
                results[camera.name]["enable"] = enable_success
        except Exception as e:
            log.error(f"Camera control error for {camera.name}: {e}")
    
    image_results = await async_capture_all_images(config)
    video_results = await async_capture_all_videos(config, duration=1)
    
    log.info("\nTest Results Summary:")
    log.info("-" * 50)
    log.info(f"Time Sync Status: {'✅' if time_sync_status else '❌'}")
    log.info("-" * 50)
    for camera_name, tests in results.items():
        multi_image = image_results.get(camera_name) is not None
        multi_video = video_results.get(camera_name) is not None
        status = "✅" if all([
            tests["single_image"],
            tests["single_video"],
            tests["disable"],
            tests["enable"],
            multi_image,
            multi_video
        ]) else "❌"
        log.info(f"Camera {camera_name} [{status}]:")
        log.info(f"  Single Image: {'✅' if tests['single_image'] else '❌'}")
        log.info(f"  Single Video: {'✅' if tests['single_video'] else '❌'}")
        log.info(f"  Disable:      {'✅' if tests['disable'] else '❌'}")
        log.info(f"  Enable:       {'✅' if tests['enable'] else '❌'}")
        log.info(f"  Multi Image:  {'✅' if multi_image else '❌'}")
        log.info(f"  Multi Video:  {'✅' if multi_video else '❌'}")
    working_cameras = sum(1 for camera_name, tests in results.items() 
                         if all([tests["single_image"], tests["single_video"],
                               tests["disable"], tests["enable"],
                               image_results.get(camera_name) is not None,
                               video_results.get(camera_name) is not None]))
    log.info("-" * 50)
    log.info(f"Working Cameras: {working_cameras}/{len(config.cameras)}")

async def main() -> None:
    parser = argparse.ArgumentParser(description="Capture images/videos from multiple cameras")
    parser.add_argument("--duration", type=int, help="Duration of video capture in seconds")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--control", choices=["enable", "disable"], help="Control camera video streams")
    parser.add_argument("--mode", choices=["image", "video"], default="image", help="Capture mode")
    parser.add_argument("--delay", type=float, default=0, help="Delay in seconds before capturing images")
    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)
    
    # Initialize config
    config = PanoramConfig()
    if args.output:
        config.output_dir = args.output
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Set up a custom thread pool executor to allow concurrent execution of blocking calls.
    loop = asyncio.get_running_loop()
    # Adjust the max_workers (here using twice the number of cameras) as needed.
    executor = ThreadPoolExecutor(max_workers=16)
    loop.set_default_executor(executor)
    
    if args.control:
        results = await async_toggle_all_cameras(config.cameras, args.control)
        success = sum(1 for success in results.values() if success)
        log.info(f"{args.control.title()}d {success}/{len(config.cameras)} cameras")
        return
    
    if args.test:
        await async_test(config)
        return
    
    duration = args.duration if args.duration is not None else config.video_duration
    
    if args.mode == "image":
        results = await async_capture_all_images(config, delay=args.delay)
    else:
        results = await async_capture_all_videos(config, duration=duration)
    
    if not all(path is not None for path in results.values()):
        log.error(f"Some {args.mode} captures failed")
        exit(1)
    
    # Print the file paths for the shell script to use
    for path in results.values():
        if path:
            print(path)

if __name__ == "__main__":
    asyncio.run(main())