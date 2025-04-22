import os
import logging
logging.basicConfig(level=logging.INFO)
from pathlib import Path

working_dir = Path.cwd()
while working_dir.name != 'gpudrive':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)

import mediapy

from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.config import EnvConfig
from gen_sim_env import GenSimGPUDriveTorchEnv

from gpudrive.env.config import EnvConfig
from examples.experimental.eval_utils import load_policy, rollout

from perception_simulation.scene_config import SceneConfig, DynamicSceneConfig, EgoConfig, StaticSceneConfig, RelightingConfig


def main():
    
    ############################## Args to parse ##############################
    data_root = "data/processed/examples"
    model_cfg_path = "baselines/ppo/config/ppo_base_sb3"
    ckpt_path = "/n/fs/pci-sharedt/mb9385/workspace/gpudrive/policy_100024842"
    device = "cpu"
    scene_config = SceneConfig(
        static_scene_config=StaticSceneConfig(
            source_path="/n/fs/pci-sharedt/jo5483/workspace/scene-generation/data/scene-generation-results/proc_geometry/waymo_fake_colmap/training/segment-102751_frontLR",
            model_path="/n/fs/pci-sharedt/aj0699/iccv25_final/iccv25_promptvariation_scenes/segment-102751_default",
            sequence_folder="/n/fs/pci-sharedt/data_processed/waymo_ns/10275144660749673822_5755_561_5775_561",
            annotation_type="waymo_annotations",
            map2scene_txt="/n/fs/pci-sharedt/data_processed/scene-generation-results/proc_geometry/waymo_surface_reconstruction/training/segment-102751/pcd/center_0-197.txt"
        ),
        dynamic_scene_config=DynamicSceneConfig(),
        ego_config=EgoConfig(
            camera_transforms_path="/n/fs/pci-sharedt/data_processed/waymo_ns/10275144660749673822_5755_561_5775_561/transforms.json",
            annotation_type="waymo_annotations",
            ego_agent_id=0  # TODO - redesign to have option to have multiple ego agents - maybe get rid of ego agent concept at all
        ),
        gaussian_type = "2D",
        save_folder="/n/fs/pci-sharedt/mb9385/workspace/gpudrive/save"
    )
    relighting_config = RelightingConfig(
        method=None,
        params={}
    )
    save_folder = "/n/fs/pci-sharedt/mb9385/workspace/gpudrive/sample_images"
    num_worlds = 1
    ###########################################################################
    
    # Dataset
    train_loader = SceneDataLoader(
        root=data_root,
        batch_size=num_worlds,
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
        scene_config=scene_config,
        relighting_config=relighting_config,
        save_folder=save_folder
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

    # Save the gif
    assert num_worlds == 1
    mediapy.write_video(os.path.join(save_folder, "bev.gif"), sim_state_frames[num_worlds], fps=15, codec='gif')


if __name__ == "__main__":
    main()
