"""Ordered LeRobot source patches; application lives in lerobot_patch_engine."""

# Patch 1: rollout context drops .eff/.ext_eff observations (see docstring).
# Patch 2: multi_task_dit.predict_action_chunk assumes select_action already
# ran _prepare_batch (stacks per-camera images into OBS_IMAGES) and
# populate_queues (n_obs_steps history). The async_inference policy server
# calls predict_action_chunk directly, so the queues are empty and it crashes
# with "stack expects a non-empty TensorList". Detect the direct call (no
# OBS_IMAGES key) and do both steps in place; the select_action path is
# unaffected because it passes an already-prepared batch.
PATCHES = [
    (
        "lerobot.rollout.context",
        ['if isinstance(v, tuple) or (v is float and k.endswith((".pos", ".vel")))'],
        'if isinstance(v, tuple) or (v is float and k.endswith((".pos", ".vel", ".eff", ".ext_eff")))',
    ),
    (
        "lerobot.policies.multi_task_dit.modeling_multi_task_dit",
        [
            # v1 of this patch: forgot to drop the ACTION/None entries the
            # server-side preprocessor leaves in the batch, so None reached
            # the queues and torch.stack raised TypeError.
            '''        self.eval()

        if self.config.image_features and OBS_IMAGES not in batch:
            # Called directly (async_inference policy server) rather than via
            # select_action: prepare the batch and fill the history queues
            # here, mirroring select_action.
            batch = self._prepare_batch(batch)
            self._queues = populate_queues(self._queues, batch)

        for k in batch:
            if k in self._queues:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)''',
            '''        self.eval()

        for k in batch:
            if k in self._queues:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)''',
            # v2 of this patch (pre-history-bundle), so re-running the
            # patcher on a v2 venv upgrades it in place.
            '''        self.eval()

        if self.config.image_features and OBS_IMAGES not in batch:
            # Called directly (async_inference policy server) rather than via
            # select_action: prepare the batch and fill the history queues
            # here, mirroring select_action (which also pops ACTION; the
            # server-side preprocessor leaves action=None in the batch).
            batch = {k: v for k, v in batch.items() if v is not None}
            batch.pop(ACTION, None)
            batch = self._prepare_batch(batch)
            self._queues = populate_queues(self._queues, batch)

        for k in batch:
            if k in self._queues:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)''',
        ],
        '''        self.eval()

        if self.config.image_features and OBS_IMAGES not in batch:
            # Called directly (async_inference policy server) rather than via
            # select_action: prepare the batch and fill the history queues
            # here, mirroring select_action (which also pops ACTION; the
            # server-side preprocessor leaves action=None in the batch).
            # tatbot patch v3: a bundled previous frame (TATBOT_OBS_HISTORY)
            # gives the n_obs=2 queues a near-spaced pair; the queue maxlen
            # evicts whatever stale cross-chunk frame was left behind.
            prev = batch.pop("tatbot_prev_batch", None)
            batch = {k: v for k, v in batch.items() if v is not None}
            batch.pop(ACTION, None)
            if prev is not None:
                prev = {k: v for k, v in prev.items() if v is not None}
                prev.pop(ACTION, None)
                prev = self._prepare_batch(prev)
                self._queues = populate_queues(self._queues, prev)
            batch = self._prepare_batch(batch)
            self._queues = populate_queues(self._queues, batch)

        for k in batch:
            if k in self._queues:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)''',
    ),
    # Patch 5: below the action-queue threshold the async client uploads full
    # raw camera frames EVERY control tick (~55 MB/s for two 640x480 RGB
    # cameras at 30 Hz). Policies with small chunks (multi_task_dit: 24) sit
    # below threshold permanently, and the sustained flood congests the
    # network segment shared with the arm's 400 Hz UDP telemetry (observed:
    # 35-79% UDP loss, driver collapse). The server consumes at most a few
    # observations per second, so cap uploads at one per 250 ms.
    (
        "lerobot.async_inference.robot_client",
        [
            '''    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold''',
        ],
        '''    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            below = self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold
        if not below:
            return False
        # tatbot patch: rate-limit uploads — raw frames at the loop rate
        # saturate the link to the server and starve arm UDP telemetry.
        now = time.monotonic()
        if now - getattr(self, "_last_obs_send_t", 0.0) < 0.25:
            return False
        self._last_obs_send_t = now
        return True''',
    ),
    # Patch 4: the async policy server allowlists policy types and predates
    # the newer policies in this release — multi_task_dit and evo1 load and
    # infer fine through its exact code path (verified on the GB10).
    (
        "lerobot.async_inference.constants",
        [
            'SUPPORTED_POLICIES = ["act", "smolvla", "diffusion", "tdmpc", "vqbet", "pi0", "pi05", "groot"]',
        ],
        'SUPPORTED_POLICIES = ["act", "smolvla", "diffusion", "tdmpc", "vqbet", "pi0", "pi05", "groot", "multi_task_dit", "evo1"]',
    ),
    # Patch 6: depth frames must be stored in millimetres. LeRobot treats
    # integer depth as mm end to end (quantization, stats, dequantized
    # output), but read_depth() returns the device's RAW units — and the
    # D405's depth unit is 0.0001 m, not the 0.001 m of most D400s. Unpatched,
    # every recorded distance is 10x too large, and real depth disagrees with
    # sim depth (true mm) by the same factor. Scale by the device's own
    # depth-scale once per connect; sub-mm precision is below both the
    # sensor's noise floor and the 12-bit codec's step at working range.
    (
        "lerobot.cameras.realsense.camera_realsense",
        [
            '''                if self.use_depth:
                    depth_frame_raw = frame.get_depth_frame()
                    depth_frame = np.asanyarray(depth_frame_raw.get_data())
                    processed_depth_frame = self._postprocess_image(depth_frame, depth_frame=True)''',
        ],
        '''                if self.use_depth:
                    depth_frame_raw = frame.get_depth_frame()
                    depth_frame = np.asanyarray(depth_frame_raw.get_data())
                    # tatbot patch: raw device units -> millimetres (see
                    # scripts/il_patch_lerobot.py patch 6; D405 = 0.1 mm/unit)
                    scale = getattr(self, "_tatbot_mm_per_unit", None)
                    if scale is None:
                        try:
                            scale = (
                                self.rs_profile.get_device()
                                .first_depth_sensor()
                                .get_depth_scale()
                                * 1000.0
                            )
                        except Exception:
                            scale = 1.0
                        self._tatbot_mm_per_unit = scale
                    if scale != 1.0:
                        depth_frame = np.rint(
                            depth_frame.astype(np.float32) * scale
                        ).astype(np.uint16)
                    processed_depth_frame = self._postprocess_image(depth_frame, depth_frame=True)''',
    ),
    # Patch 7 + 8: let multi_task_dit take wrist depth as an extra camera
    # (phase 4 of the depth plan, cheapest option first). The model stacks
    # every image feature into one (.., cam, C, H, W) tensor and validates
    # that all shapes match, so a (1, H, W) depth feature is rejected and
    # cannot stack with (3, H, W) RGB. Allow 1-channel visual features
    # through validation, and replicate them to 3 channels at the stack so
    # they ride through the CLIP encoder like any other camera. Depth is
    # already mean-std normalized per-channel from dataset stats by the
    # processor pipeline, the same treatment RGB gets.
    (
        "lerobot.policies.multi_task_dit.configuration_multi_task_dit",
        [
            '''        if len(self.image_features) > 0:
            first_key, first_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_ft.shape:
                    raise ValueError(
                        f"Image '{key}' shape {image_ft.shape} != '{first_key}' shape {first_ft.shape}"
                    )''',
        ],
        '''        if len(self.image_features) > 0:
            first_key, first_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                # tatbot patch: 1-channel (depth) features ride along with
                # RGB, replicated to 3 channels before the encoder stack.
                if image_ft.shape[1:] != first_ft.shape[1:] or image_ft.shape[0] not in (1, 3):
                    raise ValueError(
                        f"Image '{key}' shape {image_ft.shape} != '{first_key}' shape {first_ft.shape}"
                    )''',
    ),
    (
        "lerobot.policies.multi_task_dit.modeling_multi_task_dit",
        [
            '''            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)''',
        ],
        '''            # tatbot patch: replicate 1-channel depth to 3 so it stacks
            # with RGB and rides through the CLIP encoder as an extra camera.
            imgs = []
            for key in self.config.image_features:
                img = batch[key]
                if img.shape[-3] == 1:
                    img = img.expand(*img.shape[:-3], 3, *img.shape[-2:])
                imgs.append(img)
            batch[OBS_IMAGES] = torch.stack(imgs, dim=-4)''',
    ),
    # Patch 11: _reset_server() forgets last_processed_obs, so an observation
    # from the PREVIOUS client session leaks into the similarity check of the
    # next one. When the sessions' feature sets differ (an RGB-only model
    # followed by an RGBD one), make_lerobot_observation KeyErrors on the
    # missing depth key and the whole SendObservations RPC dies — the client
    # sees InactiveRpcError on every send and the arm sits at staged forever
    # (found 2026-08-24, first RGB->RGBD model switch of the squiggle eval).
    (
        "lerobot.async_inference.policy_server",
        [
            '''        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()''',
        ],
        '''        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()
        # tatbot patch: an observation from the previous client session must
        # not survive into this one — its feature set may differ (RGB vs
        # RGBD), and the similarity check KeyErrors on the mismatch.
        self.last_processed_obs = None''',
    ),
    # Patch 10: the action queue drains to EMPTY at every chunk boundary,
    # stalling the 30 Hz control loop ~0.5 s per chunk (measured 2026-08-24:
    # 21-22% of ticks lost, stall median ~480 ms, on both RGB and RGBD).
    # Two upstream halves only fail together: the client marks must_go only
    # when the queue is already empty, and the server's similarity gate
    # (joint-space atol=1 rad) drops every streamed below-threshold
    # observation during slow tool motion — so the proactive refills that
    # chunk_size_threshold is supposed to schedule never reach the policy.
    # Fire the (event-debounced) must-go at the refill threshold instead:
    # one early inference per chunk cycle, queue never empties.
    (
        "lerobot.async_inference.robot_client",
        [
            '''                observation.must_go = self.must_go.is_set() and self.action_queue.empty()''',
        ],
        '''                # tatbot patch: refill must fire BEFORE the queue empties —
                # the server similarity gate drops ordinary streamed obs during
                # slow tool motion, so this debounced must-go is the only
                # observation guaranteed to reach the policy each cycle.
                observation.must_go = self.must_go.is_set() and (
                    self.action_queue.qsize()
                    <= self._chunk_size_threshold * self.action_chunk_size
                )''',
    ),
    # Patch 9: live depth cannot reach the policy server. The async client
    # ships the follower's raw observation, so depth arrives as the camera's
    # uint16 (H, W, 1) — and the server resizes every image feature with
    # bilinear interpolate, which has no UInt16 kernel ("compute_indices_
    # weights_linear" not implemented for 'UInt16'; found on the 2026-08-24
    # depth wire bench, arm node -> serve node GPU). Cast to float32 WITHOUT
    # rescaling: the camera patch (patch 6) already put depth in millimetres
    # and the RGBD checkpoints' normalizer stats are in millimetres (median
    # ~150 mm, invalid floor 10 mm). uint8 RGB must stay uint8 — interpolate
    # supports it, and prepare_image() divides only uint8 by 255 afterwards,
    # which would be wrong for depth.
    (
        "lerobot.async_inference.helpers",
        [
            '''    dims = (resize_dims[1], resize_dims[2])
    # Add batch dimension for interpolate: (C, H, W) -> (1, C, H, W)
    image_batched = image.unsqueeze(0)''',
        ],
        '''    dims = (resize_dims[1], resize_dims[2])
    # tatbot patch: uint16 depth has no interpolate kernel — cast to float32
    # without rescaling (depth is already in millimetres, matching the
    # normalizer stats). uint8 RGB stays uint8 for prepare_image()'s /255.
    if not image.dtype.is_floating_point and image.dtype != torch.uint8:
        image = image.to(torch.float32)
    # Add batch dimension for interpolate: (C, H, W) -> (1, C, H, W)
    image_batched = image.unsqueeze(0)''',
    ),
    # Patch 21: unpickled wire images are owned NumPy arrays. Wrapping them
    # with torch.tensor copies every full-resolution camera frame before the
    # resize immediately creates another tensor. as_tensor preserves values
    # and dtype while removing that redundant host copy; the source payload is
    # not mutated and remains alive for the duration of preprocessing.
    (
        "lerobot.async_inference.helpers",
        [
            '''    image_dict = {
        key: resize_robot_observation_image(torch.tensor(lerobot_obs[key]), policy_image_features[key].shape)
        for key in image_keys
    }''',
        ],
        '''    image_dict = {
        key: resize_robot_observation_image(torch.as_tensor(lerobot_obs[key]), policy_image_features[key].shape)
        for key in image_keys
    }''',
    ),
    # Patch 12: TATBOT_OBS_HISTORY=1 bundles the previous SENT frame with
    # each observation upload. Live, the policy's n_obs=2 history frames are
    # whatever two observations last triggered inference — ~1.6 s apart at
    # chunk cadence, vs 33 ms spacing in training (the 2026-08-21 obs1
    # experiment showed that history is load-bearing for depth control).
    # The bundled pair is ~250 ms apart (the upload rate limit), 6x closer
    # to training than the status quo, not exact. OPT-IN by env flag so
    # eval-era behavior does not change silently.
    (
        "lerobot.async_inference.robot_client",
        [
            '''            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task''',
        ],
        '''            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task
            # tatbot patch: bundle the previous sent frame so the server can
            # give the policy a near-spaced 2-frame history (opt-in).
            if __import__("os").environ.get("TATBOT_OBS_HISTORY") == "1":
                _prev = getattr(self, "_tatbot_prev_raw", None)
                if _prev is not None:
                    raw_observation["_prev"] = _prev
                self._tatbot_prev_raw = {
                    k: v for k, v in raw_observation.items() if k != "_prev"
                }''',
    ),
    # Patch 13: server side of the history bundle — preprocess the attached
    # previous frame through the same pipeline and hand it to the policy as
    # tatbot_prev_batch (consumed by the patch-3 v3 queue fill).
    (
        "lerobot.async_inference.policy_server",
        [
            '''        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)''',
        ],
        '''        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        raw = observation_t.get_observation()
        # tatbot patch: an attached previous frame (see robot_client's
        # TATBOT_OBS_HISTORY bundle) becomes a near-spaced policy history
        # instead of the stale cross-chunk frame.
        prev_raw = raw.pop("_prev", None) if isinstance(raw, dict) else None
        observation: Observation = raw_observation_to_observation(
            raw,
            self.lerobot_features,
            self.policy_image_features,
        )
        _tatbot_guard_state = observation.get("observation.state")
        if _tatbot_guard_state is not None:
            _tatbot_guard_state = _tatbot_guard_state.detach().clone()
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        if prev_raw is not None:
            prev_observation = raw_observation_to_observation(
                prev_raw, self.lerobot_features, self.policy_image_features
            )
            observation = self.preprocessor(observation)
            observation["tatbot_prev_batch"] = self.preprocessor(prev_observation)
        else:
            observation = self.preprocessor(observation)''',
    ),
    # Patch 14: relative-action policies (GR00T N1.7 new embodiments) must
    # decode the WHOLE predicted chunk against the observation state that
    # produced it. Upstream async inference calls the postprocessor once per
    # timestep, losing the horizon dimension used by GR00T's native relative
    # statistics and risking each row being decoded with incompatible
    # semantics. Keep the legacy per-row path for absolute policies; use the
    # public full-chunk processor contract whenever the checkpoint declares
    # relative actions. This matches LeRobot's RTC inference path.
    (
        "lerobot.async_inference.policy_server",
        [
            # What the converged relative-actions patch above emits today: the
            # normalized-tensor clone in BOTH branches (dd7594a). Layered
            # patches must accept the current output of the layer below.
            # The pre-2026-09-09 output of this same patch, which carried the GR00T
            # noise seed. A venv patched before that still has it.
            '''        start_postprocess = time.perf_counter()
        # Opt-in no-arm diagnostic: preserve the exact normalized tensor that
        # the relative-action decoder receives. It is deliberately a log
        # record, not another wire field a robot client could consume.
        if __import__("os").environ.get("TATBOT_TRACE_PREDECODE") == "1":
            _json = __import__("json")
            self.logger.info(
                "Tatbot predecode normalized output: %s",
                _json.dumps(
                    {
                        "timestep": observation_t.get_timestep(),
                        "fixed_noise_seed": __import__("os").environ.get(
                            "TATBOT_GROOT_FIXED_NOISE_SEED"
                        ),
                        "action": action_tensor.detach().float().cpu().tolist(),
                    },
                    separators=(",", ":"),
                ),
            )
        if getattr(self.policy.config, "use_relative_actions", False):
            # tatbot patch: relative actions are one state-anchored object;
            # preserve the horizon through decode before queuing any row.
            _tatbot_normalized_action = action_tensor.detach().clone()
            action_tensor = self.postprocessor(action_tensor).squeeze(0)
        else:
            # Preserve upstream behavior for absolute-action policies whose
            # processors historically accepted only (B, action_dim).
            _tatbot_normalized_action = action_tensor.detach().clone()
            _, chunk_size, _ = action_tensor.shape
            processed_actions = []
            for i in range(chunk_size):
                single_action = action_tensor[:, i, :]
                processed_action = self.postprocessor(single_action)
                processed_actions.append(processed_action)
            action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)''',
            '''        start_postprocess = time.perf_counter()
        if getattr(self.policy.config, "use_relative_actions", False):
            # tatbot patch: relative actions are one state-anchored object;
            # preserve the horizon through decode before queuing any row.
            _tatbot_normalized_action = action_tensor.detach().clone()
            action_tensor = self.postprocessor(action_tensor).squeeze(0)
        else:
            # Preserve upstream behavior for absolute-action policies whose
            # processors historically accepted only (B, action_dim).
            _tatbot_normalized_action = action_tensor.detach().clone()
            _, chunk_size, _ = action_tensor.shape
            processed_actions = []
            for i in range(chunk_size):
                single_action = action_tensor[:, i, :]
                processed_action = self.postprocessor(single_action)
                processed_actions.append(processed_action)
            action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)''',
            # Older layer output without the clone in the relative branch.
            '''        start_postprocess = time.perf_counter()
        if getattr(self.policy.config, "use_relative_actions", False):
            # tatbot patch: relative actions are one state-anchored object;
            # preserve the horizon through decode before queuing any row.
            action_tensor = self.postprocessor(action_tensor).squeeze(0)
        else:
            # Preserve upstream behavior for absolute-action policies whose
            # processors historically accepted only (B, action_dim).
            _tatbot_normalized_action = action_tensor.detach().clone()
            _, chunk_size, _ = action_tensor.shape
            processed_actions = []
            for i in range(chunk_size):
                single_action = action_tensor[:, i, :]
                processed_action = self.postprocessor(single_action)
                processed_actions.append(processed_action)
            action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)''',
            '''        # Postprocessor expects (B, action_dim) per action, but we have (B, chunk_size, action_dim)
        # So we process each action in the chunk individually
        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape

        # Process each action in the chunk
        processed_actions = []
        for i in range(chunk_size):
            # Extract action at timestep i: (B, action_dim)
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)''',
        ],
        '''        start_postprocess = time.perf_counter()
        # Opt-in no-arm diagnostic: preserve the exact normalized tensor that
        # the relative-action decoder receives. It is deliberately a log
        # record, not another wire field a robot client could consume.
        if __import__("os").environ.get("TATBOT_TRACE_PREDECODE") == "1":
            _json = __import__("json")
            self.logger.info(
                "Tatbot predecode normalized output: %s",
                _json.dumps(
                    {
                        "timestep": observation_t.get_timestep(),
                        "action": action_tensor.detach().float().cpu().tolist(),
                    },
                    separators=(",", ":"),
                ),
            )
        if getattr(self.policy.config, "use_relative_actions", False):
            # tatbot patch: relative actions are one state-anchored object;
            # preserve the horizon through decode before queuing any row.
            _tatbot_normalized_action = action_tensor.detach().clone()
            action_tensor = self.postprocessor(action_tensor).squeeze(0)
        else:
            # Preserve upstream behavior for absolute-action policies whose
            # processors historically accepted only (B, action_dim).
            _tatbot_normalized_action = action_tensor.detach().clone()
            _, chunk_size, _ = action_tensor.shape
            processed_actions = []
            for i in range(chunk_size):
                single_action = action_tensor[:, i, :]
                processed_action = self.postprocessor(single_action)
                processed_actions.append(processed_action)
            action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)''',
    ),
    # Patch 23a: preserve the unnormalized state that anchors relative-action
    # decode. A configured live plausibility guard compares its first decoded
    # target against this exact observation before returning any actions.
    (
        "lerobot.async_inference.policy_server",
        [
            '''        observation: Observation = raw_observation_to_observation(
            raw,
            self.lerobot_features,
            self.policy_image_features,
        )
        prepare_time = time.perf_counter() - start_prepare''',
        ],
        '''        observation: Observation = raw_observation_to_observation(
            raw,
            self.lerobot_features,
            self.policy_image_features,
        )
        _tatbot_guard_state = observation.get("observation.state")
        if _tatbot_guard_state is not None:
            _tatbot_guard_state = _tatbot_guard_state.detach().clone()
        prepare_time = time.perf_counter() - start_prepare''',
    ),
    # Patch 23b: retain the normalized GR00T chunk for comparison with the
    # demo-derived normalized horizon metrics after the saved decoder runs.
    (
        "lerobot.async_inference.policy_server",
        [
            '''        if getattr(self.policy.config, "use_relative_actions", False):
            # tatbot patch: relative actions are one state-anchored object;
            # preserve the horizon through decode before queuing any row.
            action_tensor = self.postprocessor(action_tensor).squeeze(0)''',
        ],
        '''        if getattr(self.policy.config, "use_relative_actions", False):
            # tatbot patch: relative actions are one state-anchored object;
            # preserve the horizon through decode before queuing any row.
            _tatbot_normalized_action = action_tensor.detach().clone()
            action_tensor = self.postprocessor(action_tensor).squeeze(0)''',
    ),
    # Patch 23b-absolute: the async server postprocesses absolute-policy chunks
    # one row at a time. Preserve the complete normalized chunk first so ACT
    # can use the same trace path and the contract can choose whether normalized
    # metrics apply. This separate migration also upgrades already-patched venvs.
    (
        "lerobot.async_inference.policy_server",
        [
            '''        else:
            # Preserve upstream behavior for absolute-action policies whose
            # processors historically accepted only (B, action_dim).
            _, chunk_size, _ = action_tensor.shape
            processed_actions = []''',
        ],
        '''        else:
            # Preserve upstream behavior for absolute-action policies whose
            # processors historically accepted only (B, action_dim).
            _tatbot_normalized_action = action_tensor.detach().clone()
            _, chunk_size, _ = action_tensor.shape
            processed_actions = []''',
    ),
    # Patch 23c: a checkpoint-pinned server can enforce the immutable,
    # demonstration-derived trajectory contract before serializing any action.
    # The contract declares which normalized metrics apply. Absolute policies
    # such as ACT enforce the decoded-action subset; GR00T additionally checks
    # its relative min-max representation. Repeated-input spread remains an
    # offline gate because one request cannot measure it; exact-demo L1 is
    # report-only in schema 2.
    (
        "lerobot.async_inference.policy_server",
        [
            # The pre-2026-09-09 output of this same patch, which carried the GR00T
            # noise seed. A venv patched before that still has it.
            '''        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        _tatbot_trace_dir = __import__("os").environ.get("TATBOT_INFERENCE_TRACE_DIR")
        if _tatbot_trace_dir:
            if _tatbot_guard_state is None:
                raise RuntimeError("inference trace requires observation.state")
            if "_tatbot_normalized_action" not in locals():
                raise RuntimeError("inference trace requires a normalized policy chunk")
            from inference_trace import write_inference_trace

            _tatbot_trace = write_inference_trace(
                _tatbot_trace_dir,
                timestep=observation_t.get_timestep(),
                observation=raw,
                observation_state=_tatbot_guard_state,
                normalized_action=_tatbot_normalized_action,
                fixed_noise_seed=__import__("os").environ.get(
                    "TATBOT_GROOT_FIXED_NOISE_SEED"
                ),
                decoded_action=action_tensor,
            )
            self.logger.info(
                "Tatbot inference evidence: %s",
                __import__("json").dumps(_tatbot_trace, separators=(",", ":")),
            )

        _tatbot_contract_path = __import__("os").environ.get(
            "TATBOT_PLAUSIBILITY_CONTRACT"
        )
        if _tatbot_contract_path:
            if _tatbot_guard_state is None:
                raise RuntimeError("plausibility guard requires observation.state")
            from chunk_guard import enforce_chunk

            _tatbot_guard_metrics = enforce_chunk(
                _tatbot_normalized_action.detach().float().cpu().numpy(),
                action_tensor.detach().float().cpu().numpy(),
                _tatbot_guard_state.detach().float().cpu().numpy(),
                _tatbot_contract_path,
            )
            self.logger.info(
                "Tatbot chunk accepted by plausibility contract: %s",
                __import__("json").dumps(_tatbot_guard_metrics, separators=(",", ":")),
            )

        action_tensor = action_tensor.detach().cpu()''',
            '''        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        _tatbot_trace_dir = __import__("os").environ.get("TATBOT_INFERENCE_TRACE_DIR")
        if _tatbot_trace_dir:
            if _tatbot_guard_state is None:
                raise RuntimeError("inference trace requires observation.state")
            if "_tatbot_normalized_action" not in locals():
                raise RuntimeError("inference trace requires a normalized GR00T chunk")
            from inference_trace import write_inference_trace

            _tatbot_trace = write_inference_trace(
                _tatbot_trace_dir,
                timestep=observation_t.get_timestep(),
                observation=raw,
                observation_state=_tatbot_guard_state,
                normalized_action=_tatbot_normalized_action,
                decoded_action=action_tensor,
                fixed_noise_seed=__import__("os").environ.get(
                    "TATBOT_GROOT_FIXED_NOISE_SEED"
                ),
            )
            self.logger.info(
                "Tatbot inference evidence: %s",
                __import__("json").dumps(_tatbot_trace, separators=(",", ":")),
            )

        _tatbot_contract_path = __import__("os").environ.get(
            "TATBOT_PLAUSIBILITY_CONTRACT"
        )
        if _tatbot_contract_path:
            if not getattr(self.policy.config, "use_relative_actions", False):
                raise RuntimeError("plausibility guard requires relative-action policy")
            if _tatbot_guard_state is None:
                raise RuntimeError("plausibility guard requires observation.state")
            from chunk_guard import enforce_chunk

            _tatbot_guard_metrics = enforce_chunk(
                _tatbot_normalized_action.detach().float().cpu().numpy(),
                action_tensor.detach().float().cpu().numpy(),
                _tatbot_guard_state.detach().float().cpu().numpy(),
                _tatbot_contract_path,
            )
            self.logger.info(
                "Tatbot chunk passed demonstration plausibility contract: %s",
                __import__("json").dumps(_tatbot_guard_metrics, separators=(",", ":")),
            )

        action_tensor = action_tensor.detach().cpu()''',
            '''        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        action_tensor = action_tensor.detach().cpu()''',
            '''        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        _tatbot_contract_path = __import__("os").environ.get(
            "TATBOT_PLAUSIBILITY_CONTRACT"
        )
        if _tatbot_contract_path:
            if _tatbot_guard_state is None:
                raise RuntimeError("plausibility guard requires observation.state")
            from chunk_guard import enforce_chunk

            _tatbot_guard_metrics = enforce_chunk(
                _tatbot_normalized_action.detach().float().cpu().numpy(),
                action_tensor.detach().float().cpu().numpy(),
                _tatbot_guard_state.detach().float().cpu().numpy(),
                _tatbot_contract_path,
            )
            self.logger.info(
                "Tatbot chunk passed demonstration plausibility contract: %s",
                __import__("json").dumps(_tatbot_guard_metrics, separators=(",", ":")),
            )

        action_tensor = action_tensor.detach().cpu()''',
        ],
        '''        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        _tatbot_trace_dir = __import__("os").environ.get("TATBOT_INFERENCE_TRACE_DIR")
        if _tatbot_trace_dir:
            if _tatbot_guard_state is None:
                raise RuntimeError("inference trace requires observation.state")
            if "_tatbot_normalized_action" not in locals():
                raise RuntimeError("inference trace requires a normalized policy chunk")
            from inference_trace import write_inference_trace

            _tatbot_trace = write_inference_trace(
                _tatbot_trace_dir,
                timestep=observation_t.get_timestep(),
                observation=raw,
                observation_state=_tatbot_guard_state,
                normalized_action=_tatbot_normalized_action,
                decoded_action=action_tensor,
            )
            self.logger.info(
                "Tatbot inference evidence: %s",
                __import__("json").dumps(_tatbot_trace, separators=(",", ":")),
            )

        _tatbot_contract_path = __import__("os").environ.get(
            "TATBOT_PLAUSIBILITY_CONTRACT"
        )
        if _tatbot_contract_path:
            if _tatbot_guard_state is None:
                raise RuntimeError("plausibility guard requires observation.state")
            from chunk_guard import enforce_chunk

            _tatbot_guard_metrics = enforce_chunk(
                _tatbot_normalized_action.detach().float().cpu().numpy(),
                action_tensor.detach().float().cpu().numpy(),
                _tatbot_guard_state.detach().float().cpu().numpy(),
                _tatbot_contract_path,
            )
            self.logger.info(
                "Tatbot chunk accepted by plausibility contract: %s",
                __import__("json").dumps(_tatbot_guard_metrics, separators=(",", ":")),
            )

        action_tensor = action_tensor.detach().cpu()''',
    ),
    # Patch 23d: a guarded server must fail loudly as well as returning no
    # actions. Upstream turns every inference exception into an empty response,
    # which makes a client wait until its generic timeout and hides rejection.
    # Abort only when the explicit guard is configured; preserve legacy error
    # behavior for unguarded policy families.
    (
        "lerobot.async_inference.policy_server",
        [
            '''        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()''',
        ],
        '''        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")
            if __import__("os").environ.get("TATBOT_PLAUSIBILITY_CONTRACT"):
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"plausibility-guarded inference failed: {e}",
                )

            return services_pb2.Empty()''',
    ),
    # Patch 23e: schema 2 can accept a chunk with quality-review warnings, so
    # the old "passed" log text is no longer accurate. This small migration
    # also lets already-patched serving environments converge in place.
    (
        "lerobot.async_inference.policy_server",
        ['"Tatbot chunk passed demonstration plausibility contract: %s"'],
        '"Tatbot chunk accepted by plausibility contract: %s"',
    ),
    # Patch 22a: protocol 5 avoids redundant NumPy buffer copies during
    # serialization/deserialization while preserving the Python object and
    # wire contract. Python 3.12 is required across Tatbot's serving path.
    (
        "lerobot.async_inference.policy_server",
        [
            "            actions_bytes = pickle.dumps(action_chunk)  # nosec",
        ],
        "            actions_bytes = pickle.dumps(\n"
        "                action_chunk, protocol=pickle.HIGHEST_PROTOCOL\n"
        "            )  # nosec",
    ),
    # Patch 22b: use the same protocol for client setup and observation
    # payloads. Protocol 5 is readable by pickle.loads without negotiation, so
    # this changes only serialization overhead, not values or feature meaning.
    (
        "lerobot.async_inference.robot_client",
        [
            "            policy_config_bytes = pickle.dumps(self.policy_config)",
        ],
        "            policy_config_bytes = pickle.dumps(\n"
        "                self.policy_config, protocol=pickle.HIGHEST_PROTOCOL\n"
        "            )",
    ),
    (
        "lerobot.async_inference.robot_client",
        [
            "        observation_bytes = pickle.dumps(obs)",
        ],
        "        observation_bytes = pickle.dumps(obs, protocol=pickle.HIGHEST_PROTOCOL)",
    ),
    # Patch 15: an evaluation server may be deliberately pinned to one
    # checkpoint/type. Refuse a stale or mistyped client instruction instead
    # of silently loading a different model behind the same :8080 endpoint.
    # The generic server remains generic when the two env vars are unset.
    (
        "lerobot.async_inference.policy_server",
        [
            '''        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )''',
        ],
        '''        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )
        # tatbot patch: optional fail-closed deployment contract.
        _os = __import__("os")
        expected_type = _os.environ.get("TATBOT_EXPECTED_POLICY_TYPE")
        expected_path = _os.environ.get("TATBOT_EXPECTED_POLICY")
        if expected_type and policy_specs.policy_type != expected_type:
            raise ValueError(
                f"server pinned to policy type {expected_type}, got {policy_specs.policy_type}"
            )
        if expected_path:
            requested = policy_specs.pretrained_name_or_path
            same = requested == expected_path
            if _os.path.exists(requested) and _os.path.exists(expected_path):
                same = _os.path.realpath(requested) == _os.path.realpath(expected_path)
            if not same:
                raise ValueError(f"server pinned to checkpoint {expected_path}, got {requested}")''',
    ),
    # Patch 19: a checkpoint-pinned server must not allocate a second model
    # copy when the same client reconnects. Upstream loads on every
    # SendPolicyInstructions call, so a second session can OOM a 24 GB 3090
    # even though the requested policy and wire contract are unchanged.
    # Reuse only an exact policy/type/device/feature/chunk/rename match, reset
    # the policy's per-session state, and fail closed on every mismatch.
    (
        "lerobot.async_inference.policy_server",
        [
            '''        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        self.policy.to(self.device)

        # Load preprocessor and postprocessor, overriding device to match requested device
        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

        end = time.perf_counter()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")''',
        ],
        '''        requested_path = policy_specs.pretrained_name_or_path
        if _os.path.exists(requested_path):
            requested_path = _os.path.realpath(requested_path)

        if self.policy is not None:
            mismatches = []
            if getattr(self, "_tatbot_loaded_policy_path", None) != requested_path:
                mismatches.append("checkpoint")
            if self.policy_type != policy_specs.policy_type:
                mismatches.append("policy_type")
            if self.device != policy_specs.device:
                mismatches.append("device")
            if self.lerobot_features != policy_specs.lerobot_features:
                mismatches.append("features")
            if self.actions_per_chunk != policy_specs.actions_per_chunk:
                mismatches.append("actions_per_chunk")
            if getattr(self, "_tatbot_loaded_rename_map", {}) != policy_specs.rename_map:
                mismatches.append("rename_map")
            if mismatches:
                raise ValueError(
                    "loaded policy contract differs from reconnect: " + ", ".join(mismatches)
                )
            reset = getattr(self.policy, "reset", None)
            if callable(reset):
                reset()
            self.logger.info(
                "Reusing loaded checkpoint for matching client reconnect: %s", requested_path
            )
            return services_pb2.Empty()

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        self.policy.to(self.device)

        # Load preprocessor and postprocessor, overriding device to match requested device
        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )
        self._tatbot_loaded_policy_path = requested_path
        self._tatbot_loaded_rename_map = dict(policy_specs.rename_map)

        end = time.perf_counter()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")''',
    ),
    # Patch 20: ACT's ResNet backbone has an RGB-only stem, so the 1-channel
    # depth VISUAL features our fm2 datasets carry (and multi_task_dit accepts)
    # crash it with "expected input ... to have 3 channels". Expand single-
    # channel camera inputs to 3 before the backbone; RGB inputs pass through
    # untouched. First needed for the 2026-08-31 fixed-EE ACT co-train on
    # sim paper-fm2class-512 + real draw-square-fm2.
    (
        "lerobot.policies.act.modeling_act",
        [
            '''            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]''',
        ],
        '''            for img in batch[OBS_IMAGES]:
                if img.shape[1] == 1:
                    # depth cameras decode as 1-channel; ResNet stem is RGB-only
                    img = img.expand(-1, 3, -1, -1)
                cam_features = self.backbone(img)["feature_map"]''',
    ),
]
