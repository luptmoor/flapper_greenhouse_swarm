from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args

if __name__ == "__main__":

    # Parse arguments from CLI
    args = get_args()
    logger.warning("This example demonstrates the use of geometric controllers for a quadrotor.")

    # Build simulation
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="greenhouse_env",
        robot_name="flapper",
        controller_name="lee_position_control",
        args=None,
        device="cuda:0",
        num_envs=8,
        headless=args.headless,
        use_warp=args.use_warp,
    )

    # actions: reference positions in local coordinate system
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")

    # Start simulation
    env_manager.reset()

    # Simulate for 10,000 steps ~ 80s
    for i in range(10000):
        if i % 1000 == 0:
            logger.info(f"Step {i}, changing target setpoint.")
            env_manager.reset()
        env_manager.step(actions=actions)
