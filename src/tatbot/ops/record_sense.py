import os
from dataclasses import dataclass

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots import Robot, make_robot_from_config
from lerobot.robots.tatbot.config_tatbot import TatbotConfig
from PIL import Image

from tatbot.bot.urdf import get_link_poses
from tatbot.cam.depth import DepthCamera
from tatbot.cam.extrinsics import get_extrinsics
from tatbot.cam.validation import compare_extrinsics_with_urdf
from tatbot.ops.record import RecordOp, RecordOpConfig
from tatbot.utils.log import get_logger

log = get_logger("ops.sense", "🔍")

@dataclass
class SenseOpConfig(RecordOpConfig):

    num_plys: int = 2
    """Number of PLY pointcloud files to capture."""
    
    calibrate_extrinsics: bool = True
    """Whether to calibrate camera extrinsics using AprilTags."""
    
    reference_tag_id: int = 0
    """Reference AprilTag ID for extrinsics calibration."""
    
    max_deviation_warning: float = 0.05
    """Maximum deviation (m) from URDF before showing warning."""


class SenseOp(RecordOp):

    op_name: str = "sense"

    def make_robot(self) -> Robot:
        """Make a robot from the config."""
        return make_robot_from_config(
            TatbotConfig(
                ip_address_l=self.scene.arms.ip_address_l,
                ip_address_r=self.scene.arms.ip_address_r,
                arm_l_config_filepath=self.scene.arms.arm_l_config_filepath,
                arm_r_config_filepath=self.scene.arms.arm_r_config_filepath,
                goal_time=self.scene.arms.goal_time_slow,
                connection_timeout=self.scene.arms.connection_timeout,
                home_pos_l=self.scene.sleep_pos_l.joints,
                home_pos_r=self.scene.sleep_pos_r.joints,
                rs_cameras={
                    cam.name : RealSenseCameraConfig(
                        fps=cam.fps,
                        width=cam.width,
                        height=cam.height,
                        serial_number_or_name=cam.serial_number,
                    ) for cam in self.scene.cams.realsenses
                },
                ip_cameras={
                    cam.name: OpenCVCameraConfig(
                        fps=cam.fps,
                        width=cam.width,
                        height=cam.height,
                        ip=cam.ip,
                        username=cam.username,
                        password=os.environ.get(cam.password, None),
                        rtsp_port=cam.rtsp_port,
                    )
                    for cam in self.scene.cams.ipcameras
                },
            )
        )

    async def _run(self):
        _msg = "🤖 Recording observation (png images)"
        log.info(_msg)
        yield {
            'progress': 0.3,
            'message': _msg,
        }
        observation = self.robot.get_observation(full=True)
        for key, data in observation.items():
            if key in self.robot.rs_cameras or key in self.robot.ip_cameras:
                image_path = os.path.join(self.dataset_dir, f"{key}.png")
                Image.fromarray(data).save(image_path)
                log.info(f"✅ Saved frame to {image_path}")
        _msg = "✅ Saved image frames"
        log.info(_msg)
        yield {
            'progress': 0.31,
            'message': _msg,
        }
        
        _msg = "Disconnecting realsense cameras..."
        log.info(_msg)
        yield {
            'progress': 0.4,
            'message': _msg,
        }
        for cam in self.robot.rs_cameras.values():
            cam.disconnect()

        # Calibrate camera extrinsics if enabled
        if self.config.calibrate_extrinsics:
            _msg = "📐 Calibrating camera extrinsics using AprilTags..."
            log.info(_msg)
            yield {
                'progress': 0.45,
                'message': _msg,
            }
            
            # Get image paths for all cameras
            image_paths = []
            for cam in self.scene.cams.realsenses:
                image_path = os.path.join(self.dataset_dir, f"{cam.name}.png")
                if os.path.exists(image_path):
                    image_paths.append(image_path)
            for cam in self.scene.cams.ipcameras:
                image_path = os.path.join(self.dataset_dir, f"{cam.name}.png")
                if os.path.exists(image_path):
                    image_paths.append(image_path)
            
            if image_paths:
                try:
                    # Run extrinsics calibration
                    calibrated_cams = get_extrinsics(
                        image_paths=image_paths,
                        cams=self.scene.cams,
                        tags=self.scene.tags,
                    )
                    
                    # Compare with URDF positions and show warnings
                    compare_extrinsics_with_urdf(
                        calibrated_cams, 
                        self.scene, 
                        self.config.max_deviation_warning
                    )
                    
                    _msg = "✅ Camera extrinsics calibration completed"
                    log.info(_msg)
                    yield {
                        'progress': 0.48,
                        'message': _msg,
                    }
                    
                except Exception as e:
                    _msg = f"⚠️ Camera extrinsics calibration failed: {e}"
                    log.warning(_msg)
                    yield {
                        'progress': 0.48,
                        'message': _msg,
                    }
            else:
                _msg = "⚠️ No camera images found for extrinsics calibration"
                log.warning(_msg)
                yield {
                    'progress': 0.48,
                    'message': _msg,
                }

        _msg = "Connecting to Depth cameras..."
        log.info(_msg)
        yield {
            'progress': 0.5,
            'message': _msg,
        }
        link_poses = get_link_poses(
                self.scene.urdf.path, self.scene.urdf.cam_link_names, self.scene.ready_pos_full.joints
            )
        depth_cameras = {}
        for realsense in self.scene.cams.realsenses:
            depth_cameras[realsense.name] = DepthCamera(
                realsense.serial_number,
                link_poses[realsense.urdf_link_name],
                save_prefix=f"{realsense.name}_",
                save_dir=self.dataset_dir,
            )

        _msg = f"Capturing {self.config.num_plys} pointclouds..."
        log.info(_msg)
        yield {
            'progress': 0.6,
            'message': _msg,
        }
        for ply_idx in range(self.config.num_plys):
            log.info(f"Capturing pointcloud {ply_idx + 1}/{self.config.num_plys}...")
            for depth_cam in depth_cameras.values():
                depth_cam.get_pointcloud(save=True)
        
