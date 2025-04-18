import os
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

working_dir = Path.cwd()
while working_dir.name != 'gpudrive':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)

from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gen_sim_env import GenSimGPUDriveTorchEnv
import logging
logging.basicConfig(level=logging.INFO)

import dataclasses
from baselines.ppo.ppo_sb3 import load_config
from gpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv
from gpudrive.env.config import EnvConfig
from examples.experimental.eval_utils import load_policy, rollout


def main():
    
    # Args to parse
    data_root = "data/processed/examples"
    model_cfg_path = "baselines/ppo/config/ppo_base_sb3"
    ckpt_path = "/n/fs/pci-sharedt/mb9385/workspace/gpudrive/policy_100024842"
    device = "cpu"
    
    # Dataset
    train_loader = SceneDataLoader(
        root=data_root,
        batch_size=1, # Number of worlds
        dataset_size=1000,
        sample_with_replacement=False,
        shuffle=False,
    )
    print("DataLoader instantiated.")

    # Instantiate Environment
    env = GenSimGPUDriveTorchEnv(
        config=EnvConfig(),
        data_loader=train_loader,
        max_cont_agents=64, 
        device=device,
    )
    print("Env ready.")

    # Load policy
    policy = load_policy(
        path_to_cpt=ckpt_path,
        cfg_path=model_cfg_path,
        model_name="policy",
        device=device,
        env=env
    )
    print("Policy loaded.")

    obs = env.reset()[env.cont_agent_mask]

    # Show simulator to make sure we're at the same state
    env.vis.figsize = (5, 5)
    sim_states = env.vis.plot_simulator_state(
        env_indices=[0],
        zoom_radius=100,
        time_steps=[0],
    )
    ( 
        goal_achieved_count,
        frac_goal_achieved,
        collided_count,
        frac_collided,
        off_road_count,
        frac_off_road,
        not_goal_nor_crash_count,
        frac_not_goal_nor_crash_per_scene,
        controlled_agents_per_scene,
        sim_state_frames,
        agent_positions,
        episode_lengths
    ) = rollout(
        env=env, 
        policy=policy, 
        device=device, 
        render_sim_state=True,
        zoom_radius=100,
        deterministic=True,
    )

    # Evaluation
    print(f'\n Results: \n')
    print(f'Goal achieved: {frac_goal_achieved}')
    print(f'Collided: {frac_collided}')
    print(f'Off road: {frac_off_road}')
    print(f'Not goal nor crashed: {frac_not_goal_nor_crash_per_scene}')

    # Save vis
    with open("sim_state_frames.pkl", "wb") as f:
        pickle.dump(sim_state_frames, file=f)


if __name__ == "__main__":
    main()
