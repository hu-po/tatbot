"""
Captures images from all cameras and output to file.
> pip install opencv-python requests pyyaml
> python scripts/oop/camera-snapshot.py --mode image
"""

import argparse
import asyncio
from datetime import datetime
from dataclasses import dataclass
import json
import logging
import os
import time
from typing import Dict, Optional, Literal
import yaml

import cv2
import requests
from requests.auth import HTTPDigestAuth

from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

@dataclass
class Camera:
    name: str
    device_name: str
    ip: str
    MAC: str
    username: str
    password: str = None

IMAGE_FORMAT: str = "jpg"
VIDEO_FORMAT: str = "avi"
VIDEO_CODEC: str = "MJPG"
VIDEO_FPS: float = 30.0
VIDEO_DURATION: int = 5
FILENAME_TIMESTAMP_FORMAT: str = "%Yy%mm%dd%Hh%Mm%Ss"
RTSP_URL_SUFFIX: str = ":554/cam/realmonitor?channel=1&subtype=0"

CONFIG_PATH: str = os.path.expanduser("~/dev/tatbot-dev/cfg/cameras.yaml")
OUTPUT_DIR: str = os.path.expanduser("~/dev/tatbot-dev/data/snapshots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_cameras_from_config(config_filepath: str) -> Dict[str, Camera]:
    assert os.path.exists(config_filepath), f"Camera config not found: {config_filepath}"
    with open(config_filepath) as f:
        data = yaml.safe_load(f)
    cameras = {}
    for key, value in data.items():
        value['name'] = key
        camera = Camera(**value)
        cameras[key] = camera
        cameras[key].password = os.getenv(f"{str.upper(key)}_PASSWORD")
    return cameras

# Initially load cameras. (This global will be updated in main() if needed.)
CAMERAS: Dict[str, Camera] = load_cameras_from_config(CONFIG_PATH)

def make_rtsp_url(camera: Camera) -> str:
    return f"rtsp://{camera.username}:{camera.password}@{camera.ip}{RTSP_URL_SUFFIX}"

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

def capture_image(camera: Camera, file_suffix: str, output_dir: str, filename_timestamp_format: str, delay: float = 0) -> Optional[str]:
    cap = cv2.VideoCapture(make_rtsp_url(camera))
    if not cap.isOpened():
        log.error(f"Could not open stream for {camera.name} at {make_rtsp_url(camera)}")
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
    filename = f"{timestamp}_{file_suffix}.{IMAGE_FORMAT}"
    filepath = os.path.join(output_dir, filename)
    
    cv2.imwrite(filepath, frame)
    log.info(f"Saved image from {camera.name} ({file_suffix}) to {filepath}")
    return filepath

def capture_video(camera: Camera, file_suffix: str, duration: int, output_dir: str, filename_timestamp_format: str, video_codec: str, video_fps: float) -> Optional[str]:
    cap = cv2.VideoCapture(make_rtsp_url(camera))
    if not cap.isOpened():
        log.error(f"Could not open stream for {camera.name} at {make_rtsp_url(camera)}")
        return None

    ret, frame = cap.read()
    if not ret or frame is None:
        log.error(f"Failed to read initial frame for {camera.name}")
        cap.release()
        return None

    height, width = frame.shape[:2]
    timestamp = datetime.now().strftime(filename_timestamp_format)
    filename = f"{timestamp}_{file_suffix}.avi"
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

async def async_get_image(camera: Camera, file_suffix: str, delay: float = 0) -> Optional[str]:
    return await asyncio.to_thread(capture_image, camera, file_suffix, OUTPUT_DIR, FILENAME_TIMESTAMP_FORMAT, delay)

async def async_get_video(camera: Camera, file_suffix: str, duration: int = VIDEO_DURATION) -> Optional[str]:
    return await asyncio.to_thread(capture_video, camera, file_suffix, duration, OUTPUT_DIR, FILENAME_TIMESTAMP_FORMAT, VIDEO_CODEC, VIDEO_FPS)

async def async_capture_all_images(delay: float = 0) -> Dict[str, Optional[str]]:
    tasks = {
        camera_id: asyncio.create_task(async_get_image(camera, f"multi_{camera.name}", delay))
        for camera_id, camera in CAMERAS.items()
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

async def async_capture_all_videos(duration: int = VIDEO_DURATION) -> Dict[str, Optional[str]]:
    tasks = {
        camera_id: asyncio.create_task(async_get_video(camera, f"multi_{camera.name}", duration))
        for camera_id, camera in CAMERAS.items()
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

async def async_test() -> None:
    log.info("Running tests...")
    log.setLevel(logging.DEBUG)
    results: Dict[str, Dict[str, bool]] = {}
    
    log.info("\nVerifying camera times...")
    time_sync_status = verify_all_cameras_same_time(CAMERAS)
    
    for camera in CAMERAS.values():
        results[camera.name] = {
            "single_image": False,
            "single_video": False,
            "disable": False,
            "enable": False
        }
        try:
            image_path = await async_get_image(camera, "test_single")
            results[camera.name]["single_image"] = image_path is not None
        except Exception as e:
            log.error(f"Single image capture error for {camera.name}: {e}")
        
        try:
            video_path = await async_get_video(camera, "test_single", duration=1)
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
    
    image_results = await async_capture_all_images()
    video_results = await async_capture_all_videos(duration=1)
    
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
    log.info(f"Working Cameras: {working_cameras}/{len(CAMERAS)}")

async def main() -> None:
    global OUTPUT_DIR, CAMERAS
    
    parser = argparse.ArgumentParser(description="Capture images/videos from multiple cameras")
    parser.add_argument("--duration", type=int, default=VIDEO_DURATION, help="Duration of video capture in seconds")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--control", choices=["enable", "disable"], help="Control camera video streams")
    parser.add_argument("--mode", choices=["image", "video"], default="image", help="Capture mode")
    parser.add_argument("--delay", type=float, default=0, help="Delay in seconds before capturing images")
    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)
    
    OUTPUT_DIR = args.output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set up a custom thread pool executor to allow concurrent execution of blocking calls.
    loop = asyncio.get_running_loop()
    # Adjust the max_workers (here using twice the number of cameras) as needed.
    executor = ThreadPoolExecutor(max_workers=16)
    loop.set_default_executor(executor)
    
    cameras = load_cameras_from_config(CONFIG_PATH)
    # Update the global CAMERAS so that other functions use the newly loaded config.
    CAMERAS = cameras
    
    if args.control:
        results = await async_toggle_all_cameras(cameras, args.control)
        success = sum(1 for success in results.values() if success)
        log.info(f"{args.control.title()}d {success}/{len(cameras)} cameras")
        return
    
    if args.test:
        await async_test()
        return
    
    if args.mode == "image":
        results = await async_capture_all_images(delay=args.delay)
    else:
        results = await async_capture_all_videos(duration=args.duration)
    
    if not all(path is not None for path in results.values()):
        log.error(f"Some {args.mode} captures failed")
        exit(1)
    
    # Print the file paths for the shell script to use
    for path in results.values():
        if path:
            print(path)

if __name__ == "__main__":
    asyncio.run(main())