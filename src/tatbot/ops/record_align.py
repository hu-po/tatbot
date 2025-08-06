import os
import time
from dataclasses import dataclass

from lerobot.datasets.utils import build_dataset_frame
from lerobot.utils.robot_utils import busy_wait

from tatbot.data.stroke import StrokeBatch, StrokeList
from tatbot.gen.align import make_align_strokes
from tatbot.gen.batch import strokebatch_from_strokes
from tatbot.mcp.gpu_proxy import GPUProxy, check_local_gpu
from tatbot.ops.record import RecordOp, RecordOpConfig
from tatbot.utils.log import get_logger

log = get_logger("ops.align", "📐")


@dataclass
class AlignOpConfig(RecordOpConfig):
    pass


class AlignOp(RecordOp):

    op_name: str = "align"

    async def _run(self):
        _msg = "Generating alignment strokes..."
        log.info(_msg)
        yield {
            'progress': 0.2,
            'message': _msg,
        }
        
        try:
            strokes: StrokeList = make_align_strokes(self.scene)
            log.info(f"✅ Generated {len(strokes.strokes)} alignment stroke pairs")
        except Exception as e:
            log.error(f"❌ Error generating alignment strokes: {e}")
            raise
        
        strokes_path = os.path.join(self.dataset_dir, "strokes.yaml")
        strokebatch_path = os.path.join(self.dataset_dir, "strokebatch.safetensors")
        
        try:
            log.info(f"💾 Saving strokes to {strokes_path}")
            strokes.to_yaml_with_arrays(strokes_path)
            
            # Verify the file was actually created
            import os
            if os.path.exists(strokes_path):
                file_size = os.path.getsize(strokes_path)
                log.info(f"✅ strokes.yaml created successfully ({file_size} bytes)")
            else:
                raise FileNotFoundError(f"strokes.yaml was not created at {strokes_path}")
            
            # Give NFS a moment to sync files before remote call
            import time
            log.info("⏳ Waiting for NFS sync before remote GPU call...")
            time.sleep(1.0)
        except Exception as e:
            log.error(f"❌ Error in strokes.to_yaml_with_arrays: {e}")
            raise
        
        # Check if we need to use remote GPU for conversion
        if check_local_gpu():
            log.info("Using local GPU for strokebatch conversion")
            strokebatch: StrokeBatch = strokebatch_from_strokes(self.scene, strokes, first_last_rest=False)
            strokebatch.save(strokebatch_path)
        else:
            log.info("Using remote GPU node for strokebatch conversion")
            gpu_proxy = GPUProxy()
            
            # Since all nodes share NFS, just pass the file path instead of YAML content
            success, _ = await gpu_proxy.convert_strokelist_remote(
                strokes_file_path=strokes_path,
                strokebatch_file_path=strokebatch_path,
                scene_name=self.scene.name,
                first_last_rest=False,
                use_ee_offsets=True
            )
            
            if not success:
                raise RuntimeError("Failed to convert strokes to strokebatch on remote GPU node")
            
            # File is already saved to NFS by the remote GPU node
            
            # Load it for use
            strokebatch = StrokeBatch.load(strokebatch_path)
            
        log.info(f"Strokebatch created with shape: {strokebatch.joints.shape}")

        # maximally retracted when performing alignment operation
        offset_idx_l = self.scene.arms.offset_num - 1
        offset_idx_r = self.scene.arms.offset_num - 1

        for stroke_idx, (stroke_l, stroke_r) in enumerate(strokes.strokes):

            # make sure robot is connected and in ready position
            if not self.robot.is_connected:
                log.warning("⚠️ Robot is not connected. Attempting to reconnect...")
                self.robot.connect()
                if not self.robot.is_connected:
                    raise RuntimeError("❌ Failed to connect to robot")
            self.robot.send_action(self.robot._urdf_joints_to_action(self.scene.ready_pos_full.joints), safe=True)

            _msg = f"🔍 Executing stroke {stroke_idx + 1}/{len(strokes.strokes)}: left={stroke_l.description}, right={stroke_r.description}"
            log.info(_msg)
            yield {
                'progress': 0.3 + (0.6 * stroke_idx / len(strokes.strokes)),
                'message': _msg,
            }

            for pose_idx in range(self.scene.stroke_length):
                start_loop_t = time.perf_counter()
                
                observation = self.robot.get_observation()
                log.info(f"observation: {observation}")
                log.info(f"dataset features: {self.dataset.features}")
                observation_frame = build_dataset_frame(self.dataset.features, observation, prefix="observation")
                
                joints = strokebatch.offset_joints(stroke_idx, pose_idx, offset_idx_l, offset_idx_r)
                robot_action = self.robot._urdf_joints_to_action(joints)
                if pose_idx == 0 or pose_idx == self.scene.stroke_length - 1:
                    # use slow movements for first and last poses
                    sent_action = self.robot.send_action(robot_action, self.scene.arms.goal_time_slow, safe=True)
                else:
                    sent_action = self.robot.send_action(robot_action, self.scene.arms.goal_time_fast)

                action_frame = build_dataset_frame(self.dataset.features, sent_action, prefix="action")
                frame = {**observation_frame, **action_frame}
                self.dataset.add_frame(frame, task=f"left: {stroke_l.description}, right: {stroke_r.description}")

                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / self.config.fps - dt_s)

            self.dataset.save_episode()