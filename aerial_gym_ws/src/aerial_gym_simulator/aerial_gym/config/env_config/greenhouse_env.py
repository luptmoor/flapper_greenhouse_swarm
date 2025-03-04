from aerial_gym.config.asset_config.env_object_config import bottom_wall, left_wall, right_wall, back_wall, front_wall, tree_asset_params, capsicum

class GreenhouseEnvironmentCfg:
    class env:
        num_envs = 0  # number of environments
        num_env_actions = 0  # this is the number of actions handled by the environment
        # these are the actions that are sent to environment entities
        # and some of them may be used to control various entities in the environment
        # e.g. motion of obstacles, etc.
        env_spacing = 1.0  # not used with heightfields/trimeshes
        num_physics_steps_per_env_step_mean = 1  # number of steps between camera renders mean
        num_physics_steps_per_env_step_std = 0  # number of steps between camera renders std
        render_viewer_every_n_steps = 10  # render the viewer every n steps
        collision_force_threshold = 0.010  # collision force threshold
        manual_camera_trigger = False  # trigger camera captures manually
        reset_on_collision = (
            True  # reset environment when contact force on quadrotor is above a threshold
        )
        create_ground_plane = False  # create a ground plane
        sample_timestep_for_latency = True  # sample the timestep for the latency noise
        perturb_observations = True
        keep_same_env_for_num_episodes = 1
        write_to_sim_at_every_timestep = False  # write to sim at every timestep

        use_warp = False
        e_s = env_spacing
        lower_bound_min = [-e_s, -e_s, -e_s]  # lower bound for the environment space
        lower_bound_max = [-e_s, -e_s, -e_s]  # lower bound for the environment space
        upper_bound_min = [e_s, e_s, e_s]  # upper bound for the environment space
        upper_bound_max = [e_s, e_s, e_s]  # upper bound for the environment space

    class env_config:
        include_asset_type = {
            "bottom_wall": True,
            "front_wall": False,
            "back_wall": False,
            "left_wall": False,
            "right_wall": False,
            "tree": False,
            "capsicum": True
        }

        asset_type_to_dict_map = {
            "bottom_wall": bottom_wall,
            "front_wall": front_wall,
            "back_wall": back_wall,
            "left_wall": left_wall,
            "right_wall": right_wall,
            "tree": tree_asset_params,
            "capsicum": capsicum
        }
