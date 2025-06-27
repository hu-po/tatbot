# combination on gen, ik, and plan


@jdc.pytree_dataclass
class PathBatch:
    ee_pos_l: Float[Array, "b l 3"]
    """End effector frame position in meters (x, y, z) for left arm."""
    ee_pos_r: Float[Array, "b l 3"]
    """End effector frame position in meters (x, y, z) for right arm."""
    ee_wxyz_l: Float[Array, "b l 4"]
    """End effector frame orientation as quaternion (w, x, y, z) for left arm."""
    ee_wxyz_r: Float[Array, "b l 4"]
    """End effector frame orientation as quaternion (w, x, y, z) for right arm."""
    joints: Float[Array, "b l 16"]
    """Joint positions in radians (URDF convention)."""
    dt: Float[Array, "b l"]
    """Travel time from pose N to pose N+1 in seconds."""

    @classmethod
    def empty(cls, length: int) -> "PathBatch":
        log.debug(f"🔳 Creating empty pathbatch of length {length}...")
        return cls(
            ee_pos_l=jnp.zeros((1, length, 3), dtype=jnp.float32),
            ee_pos_r=jnp.zeros((1, length, 3), dtype=jnp.float32),
            ee_wxyz_l=jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32), (1, length, 1)),
            ee_wxyz_r=jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32), (1, length, 1)),
            joints=jnp.zeros((1, length, 16), dtype=jnp.float32),
            dt=jnp.full((1, length,), 0.01, dtype=jnp.float32), # seconds
        )
    
    def add_strokes(self, strokes: list[Stroke], image: Image):
        log.debug(f"⚙️ Input image shape: {image.size}")
        self.save_image_np(image)
        self.image_width_px = image.size[0]
        self.image_height_px = image.size[1]
        scale_x = self.image_width_m / self.image_width_px
        scale_y = self.image_height_m / self.image_height_px

        log.info(f"⚙️ Adding {len(strokes)} raw paths to plan...")
        for idx, stroke in enumerate(strokes):
            if stroke.pixel_coords is not None:
                stroke.pixel_coords = np.array(stroke.pixel_coords, dtype=int)
            stroke_length = len(stroke.pixel_coords)
            desired_length = self.path_length - 2 # -2 for hover positions
            if stroke_length != desired_length:
                log.warning(f"⚙️⚠️ stroke {idx} has len {stroke_length}, resampling to {desired_length}...")
                if stroke_length > 1:
                    # Calculate the cumulative distance along the path
                    distances = np.sqrt(np.sum(np.diff(stroke.pixel_coords, axis=0)**2, axis=1))
                    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
                    
                    # Create a new set of evenly spaced distances
                    new_distances = np.linspace(0, cumulative_distances[-1], desired_length)
                    
                    # Interpolate the x and y coordinates
                    x_coords_resampled = np.interp(new_distances, cumulative_distances, stroke.pixel_coords[:, 0])
                    y_coords_resampled = np.interp(new_distances, cumulative_distances, stroke.pixel_coords[:, 1])
                    
                    stroke.pixel_coords = np.round(np.stack((x_coords_resampled, y_coords_resampled), axis=-1)).astype(int)
                elif stroke_length == 1:
                    stroke.pixel_coords = np.tile(stroke.pixel_coords, (desired_length, 1))
                else: # stroke_length == 0
                    stroke.pixel_coords = np.zeros((desired_length, 2), dtype=int)
                stroke_length = desired_length
                
            # add normalized coordinates: top left is 0, 0
            stroke.norm_coords = stroke.pixel_coords / np.array([self.image_width_px, self.image_height_px], dtype=np.float32)
            # calculate center of mass of stroke
            stroke.norm_center = np.mean(stroke.norm_coords, axis=0)
            # calculate meters coordinates: center is 0, 0
            meter_coords_2d = (
                stroke.pixel_coords * np.array([scale_x, scale_y], dtype=np.float32)
                - np.array([self.image_width_m / 2, self.image_height_m / 2], dtype=np.float32)
            )
            stroke.meter_coords = np.hstack([
                meter_coords_2d,
                np.zeros((stroke_length, 1), dtype=np.float32)
            ])
            # calculate center of mass of stroke
            stroke.meters_center = np.mean(stroke.meter_coords, axis=0)
            self.strokes[f'stroke_{idx:03d}'] = stroke

        self.save() # update metadata
        self.calculate_pathbatch()

    def make_inkdip_pos(self, inkcap_name: str) -> np.ndarray:
        assert inkcap_name in self.ink_config.inkcaps, f"⚙️❌ Inkcap {inkcap_name} not found in palette"
        inkcap: InkCap = self.ink_config.inkcaps[inkcap_name]
        inkcap_pos = np.array([0, 0, 0], dtype=np.float32)
        # hover over the inkcap
        inkdip_pos = transform_and_offset(
            np.tile(inkcap_pos, (self.path_length, 1)),
            self.ink_config.inkpalette_pos,
            self.ink_config.inkpalette_wxyz,
            self.ink_config.inkdip_hover_offset,
        )
        # Split: 1/3 down, 1/3 wait, 1/3 up (adjust as needed)
        num_down = self.path_length // 3
        num_up = self.path_length // 3
        num_wait = self.path_length - num_down - num_up
        # dip down to inkcap depth
        down_z = np.linspace(0, inkcap.depth_m, num_down, endpoint=False)
        # wait at depth
        wait_z = np.full(num_wait, inkcap.depth_m)
        # retract back up
        up_z = np.linspace(inkcap.depth_m, 0, num_up, endpoint=True)
        # concatenate into offset array
        offsets = np.hstack([
            np.zeros((self.path_length, 2)),
            -np.concatenate([down_z, wait_z, up_z]).reshape(-1, 1),
        ])
        inkdip_pos = transform_and_offset(
            inkdip_pos,
            self.ink_config.inkpalette_pos,
            self.ink_config.inkpalette_wxyz,
            offsets,
        )
        return inkdip_pos

    def calculate_pathbatch(self, filter_completed: bool = False) -> None:
        # paths will be accumulated then concatenated into a pathbatch object
        paths: list[Path] = []

        if filter_completed:
            # if pathbatch is re-calculated halfway through the session
            log.debug(f"⚙️ Filtering out completed strokes...")
            _strokes = {k: v for k, v in self.strokes.items() if v.is_completed}
        else:
            _strokes = self.strokes

        # sort strokes along the -Y axis in normalized coords
        sorted_strokes = sorted(_strokes.items(), key=lambda x: x[1].norm_center[1], reverse=True)
        left_arm_pointer = 0 # left arm starts with leftmost stroke, moves rightwards
        half_length = len(sorted_strokes) // 2
        right_arm_pointer = half_length # right arm starts from middle moves rightwards
        total_strokes = len(sorted_strokes)
        while left_arm_pointer <= half_length and right_arm_pointer < total_strokes:
            stroke_name_l, stroke_l = sorted_strokes[left_arm_pointer]
            assert stroke_name_l in self.strokes, f"⚙️❌ Stroke {stroke_name_l} not found in plan"
            stroke_name_r, stroke_r = sorted_strokes[right_arm_pointer]
            assert stroke_name_r in self.strokes, f"⚙️❌ Stroke {stroke_name_r} not found in plan"
            log.debug(f"⚙️ Building path from strokes left: {stroke_name_l} and right: {stroke_name_r}...")
            
            # create a new empty path for this stroke
            path = Path.empty(self.path_length)
            # set the default time between poses to fast movement
            path.dt[:] = self.path_dt_fast
            # slow movement to and from hover positions
            path.dt[:2] = self.path_dt_slow
            path.dt[-2:] = self.path_dt_slow
            # TODO: for now orientation is just design orientation (for inkdips as well)
            path.ee_wxyz_l[:, :] = np.tile(self.ee_wxyz_l, (self.path_length, 1))
            path.ee_wxyz_r[:, :] = np.tile(self.ee_wxyz_r, (self.path_length, 1))

            # left arm pointer hits a stroke with no inkcap
            if self.strokes[stroke_name_l].inkcap is None:
                # get ink color from stroke, left arm will dip for this path
                inkcap_name = self.ink_config.find_best_inkcap(self.strokes[stroke_name_l].color)
                self.strokes[stroke_name_l].inkcap = inkcap_name
                path.ee_pos_l = self.make_inkdip_pos(inkcap_name)
                # make a new stroke object for the inkdip path
                stroke_l = Stroke(
                    description=f"left arm inkdip into {inkcap_name}",
                    is_inkdip=True,
                    inkcap=inkcap_name,
                )
                if left_arm_pointer == 0:
                    # this is the first stroke of the session, keep right arm at rest for this path
                    stroke_r = Stroke(description="right arm rest")
                    self.path_idx_to_strokes.append([stroke_l, stroke_r])
                    paths.append(path)
                    continue
            elif left_arm_pointer < half_length: # still haven't hit the last left-arm stroke
                # left arm pointer hits a stroke with an inkcap
                # transform to design frame, add needle offset
                path.ee_pos_l[1:-1, :] = transform_and_offset(
                    self.strokes[stroke_name_l].meter_coords,
                    self.design_pos,
                    self.design_wxyz,
                    self.needle_offset_l,
                )
                # add hover positions to start and end
                path.ee_pos_l[0, :] = transform_and_offset(
                    np.expand_dims(self.strokes[stroke_name_l].meter_coords[0], axis=0),
                    self.design_pos,
                    self.design_wxyz,
                    self.hover_offset,
                )
                path.ee_pos_l[-1, :] = transform_and_offset(
                    np.expand_dims(self.strokes[stroke_name_l].meter_coords[-1], axis=0),
                    self.design_pos,
                    self.design_wxyz,
                    self.hover_offset,
                )
                left_arm_pointer += 1
            else:
                stroke_l = Stroke(description="left arm rest")

            if self.strokes[stroke_name_r].inkcap is None:
                inkcap_name = self.ink_config.find_best_inkcap(self.strokes[stroke_name_r].color)
                self.strokes[stroke_name_r].inkcap = inkcap_name
                path.ee_pos_r = self.make_inkdip_pos(inkcap_name)
                # make a new stroke object for the inkdip path
                stroke_r = Stroke(
                    description=f"right arm inkdip into {inkcap_name}",
                    is_inkdip=True,
                    inkcap=inkcap_name,
                )
            else:
                # right arm pointer hits a stroke with an inkcap
                # transform to design frame, add needle offset
                path.ee_pos_r[1:-1, :] = transform_and_offset(
                    self.strokes[stroke_name_r].meter_coords,
                    self.design_pos,
                    self.design_wxyz,
                    self.needle_offset_r,
                )
                # add hover positions to start and end
                path.ee_pos_r[0, :] = transform_and_offset(
                    np.expand_dims(self.strokes[stroke_name_r].meter_coords[0], axis=0),
                    self.design_pos,
                    self.design_wxyz,
                    self.hover_offset,
                )
                path.ee_pos_r[-1, :] = transform_and_offset(
                    np.expand_dims(self.strokes[stroke_name_r].meter_coords[-1], axis=0),
                    self.design_pos,
                    self.design_wxyz,
                    self.hover_offset,
                )
                right_arm_pointer += 1

            stroke_l.arm = "left"
            stroke_r.arm = "right"
            log.debug(f"⚙️ Adding path from strokes\n left: {stroke_l.description}\n right: {stroke_r.description}...")
            self.path_idx_to_strokes.append([stroke_l, stroke_r])
            paths.append(path)

        # HACK: add two paths at the beginning where left arm is centered over design and right arm is centered over black inkcap
        # use twice the hover offset to ensure enough space for adjustments
        path = Path.empty(self.path_length)
        path.ee_pos_l = transform_and_offset(
            np.zeros((self.path_length, 3)),
            self.design_pos,
            self.design_wxyz,
            self.hover_offset * 2,
        )
        path.ee_wxyz_l = np.tile(self.ee_wxyz_l, (self.path_length, 1))
        path.ee_pos_r = transform_and_offset(
            np.tile([0, 0, 0], (self.path_length, 1)),
            self.ink_config.inkpalette_pos,
            self.ink_config.inkpalette_wxyz,
            self.ink_config.inkdip_hover_offset * 2,
        )
        path.ee_wxyz_r = np.tile(self.ee_wxyz_r, (self.path_length, 1))
        path.dt[:2] = self.path_dt_slow
        path.dt[-2:] = self.path_dt_slow
        paths.insert(0, path)
        stroke_l = Stroke(description="left over design with twice the hover offset")
        stroke_r = Stroke(description="right over large black inkcap with twice the hover offset")
        self.path_idx_to_strokes.insert(0, [stroke_l, stroke_r])

        # HACK: second hack path
        path = Path.empty(self.path_length)
        path.ee_pos_l = transform_and_offset(
            np.tile([0, 0, 0], (self.path_length, 1)),
            self.ink_config.inkpalette_pos,
            self.ink_config.inkpalette_wxyz,
            self.ink_config.inkdip_hover_offset * 2,
        )
        path.ee_wxyz_l = np.tile(self.ee_wxyz_l, (self.path_length, 1))
        path.ee_pos_r = transform_and_offset(
            np.zeros((self.path_length, 3)),
            self.design_pos,
            self.design_wxyz,
            self.hover_offset * 2,
        )
        path.ee_wxyz_r = np.tile(self.ee_wxyz_r, (self.path_length, 1))
        path.dt[:2] = self.path_dt_slow
        path.dt[-2:] = self.path_dt_slow
        paths.insert(0, path)
        stroke_l = Stroke(description="left over large black inkcap with twice the hover offset")
        stroke_r = Stroke(description="right over design with twice the hover offset")
        self.path_idx_to_strokes.insert(0, [stroke_l, stroke_r])

        # Perform IK in batches (batch size will be hardware specific)
        flat_target_pos   : list[list[np.ndarray]] = []
        flat_target_wxyz  : list[list[np.ndarray]] = []
        index_map: list[tuple[int, int]] = [] # (path_idx, pose_idx)
        for p_idx, path in enumerate(paths):
            for pose_idx in range(path.ee_pos_l.shape[0]):
                index_map.append((p_idx, pose_idx))
                flat_target_pos.append(
                    [path.ee_pos_l[pose_idx], path.ee_pos_r[pose_idx]]
                )
                flat_target_wxyz.append(
                    [path.ee_wxyz_l[pose_idx], path.ee_wxyz_r[pose_idx]]
                )
        target_pos   = jnp.array(flat_target_pos)    # (B, 2, 3)
        target_wxyz  = jnp.array(flat_target_wxyz)   # (B, 2, 4)
        for start in range(0, target_pos.shape[0], self.ik_batch_size):
            end = start + self.ik_batch_size
            batch_pos   = target_pos[start:end]       # (b, 2, 3)
            batch_wxyz  = target_wxyz[start:end]      # (b, 2, 4)
            batch_joints = batch_ik(
                target_wxyz=batch_wxyz,
                target_pos=batch_pos,
            )                                         # (b, 16)
            # write results back into the corresponding path / pose slots
            for local_idx, joints in enumerate(batch_joints):
                p_idx, pose_idx = index_map[start + local_idx]
                paths[p_idx].joints[pose_idx] = np.asarray(joints, dtype=np.float32)

        # HACK: the right arm of the first (not counting hack paths) path should be at rest while left arm is ink dipping
        paths[2].joints[:, 8:] = np.tile(BotConfig().rest_pose[8:], (self.path_length, 1))
        # HACK: the left arm of the final path should be at rest since last stroke is right-only
        paths[-1].joints[:, :8] = np.tile(BotConfig().rest_pose[:8], (self.path_length, 1))

        pathbatch = PathBatch.from_paths(paths)
        self.save_pathbatch(pathbatch)
        self.save() # update metadata

def plan_from_svg(config: SVGPlanConfig):
    plan = Plan(
        name="svg",
        dirpath=config.output_dir,
    )

    log.info(f"🖋️📂 Loading SVG file from {config.svg_path}")
    svg_path = os.path.expanduser(config.svg_path)
    paths, attributes, svg_attr = svgpathtools.svg2paths2(svg_path)

    strokes: list[Stroke] = []
    for i, path in enumerate(paths):
        if path.length() == 0:
            continue

        ts = np.linspace(0, 1, config.points_per_path)
        points = np.array([path.point(t) for t in ts])
        pixel_coords = np.round(np.column_stack((points.real, points.imag))).astype(int)

        stroke = Stroke(
            pixel_coords=pixel_coords,
            description=f"svg_path_{i}",
            color="black"
        )
        strokes.append(stroke)

    log.info(f"🖋️ Generated {len(strokes)} strokes.")

    if config.png_path is not None:
        image = Image.open(os.path.expanduser(config.png_path))
        plan.image_width_px = image.width
        plan.image_height_px = image.height

    plan.add_strokes(strokes, image)