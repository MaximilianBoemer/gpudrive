
import numpy as np
import torch

from gpudrive.datatypes.observation import GlobalEgoState
from gpudrive.datatypes.roadgraph import GlobalRoadGraphPoints
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.config import RenderConfig

from perception_simulation.scene_config import SceneConfig, DynamicSceneConfig, EgoConfig, StaticSceneConfig, RelightingConfig
from perception_simulation.perception_sim import Scene


class GenSimGPUDriveTorchEnv(GPUDriveTorchEnv):
    """Torch Gym Environment that interfaces with the GPU Drive simulator including GenSim integration."""
    def __init__(
        self,
        config,
        data_loader,
        max_cont_agents,
        device="cuda",
        action_type="discrete",
        render_config: RenderConfig = RenderConfig(),
        backend="torch",
    ):
        super().__init__(
            config=config,
            data_loader=data_loader,
            max_cont_agents=max_cont_agents,
            device=device,
            action_type=action_type,
            render_config=render_config,
            backend=backend,
        )
        assert data_loader.batch_size == 1, "Number of worlds must be 1 for GenSim integration for now."

        # TODO - move out of here and have it as an arg
        scene_config = SceneConfig(
            static_scene_config=StaticSceneConfig(
                source_path="/n/fs/pci-sharedt/data_processed/scene-generation-results/proc_geometry/xcube_fake_colmap/segment-10275144660749673822_5755_561_5775_561_with_camera_labels",
                model_path="/n/fs/pci-sharedt/aj0699/iccv25_final/iccv25_promptvariation_scenes/segment-102751_default",
                sequence_folder="/n/fs/pci-sharedt/data_processed/waymo_ns/10275144660749673822_5755_561_5775_561",
                annotation_type="waymo_annotations",
                map2scene_txt="/n/fs/pci-sharedt/data_processed/scene-generation-results/proc_geometry/waymo_surface_reconstruction/training/segment-102751/pcd/center_0-197.txt"
            ),
            dynamic_scene_config=DynamicSceneConfig(),
            ego_config=EgoConfig(
                camera_transforms_path="/n/fs/pci-sharedt/data_processed/waymo_ns/10275144660749673822_5755_561_5775_561/transforms.json",
                annotation_type="waymo_annotations",
                ego_agent_id=0
            ),
            gaussian_type = "2D",
            save_folder="/n/fs/pci-sharedt/mb9385/workspace/gpudrive/save"
        )
        relighting_config = RelightingConfig(
            method=None,
            params={}
        )
        self.gen_sim_scene = Scene(
            scene_config=scene_config,
            relighting_config=relighting_config
        )
        print("")


    def get_obs(self, mask=None):
        """Get observation: Combine different types of environment information into a single tensor.

        Returns:
            torch.Tensor: (num_worlds, max_agent_count, num_features)
        """
        ego_states = self._get_ego_state(mask)
        partner_observations = self._get_partner_obs(mask)
        road_map_observations = self._get_road_map_obs(mask)
        lidar_observations = self._get_lidar_obs(mask)
        image_observations = self._get_image_obs(mask)

        obs = torch.cat(
            (
                ego_states,
                partner_observations,
                road_map_observations,
            ),
            dim=-1,
        )

        return obs
    
    def _get_image_obs(self, mask=None):
        agent_state = GlobalEgoState.from_tensor(
            self.sim.absolute_self_observation_tensor(),
            self.backend,
            device=self.device,
        )
        global_roadgraph = GlobalRoadGraphPoints.from_tensor(
            roadgraph_tensor=self.sim.map_observation_tensor(),
            backend=self.backend,
            device=self.device,
        )
        
        print("")
        import os
        save_folder = "/n/fs/pci-sharedt/mb9385/workspace/gpudrive/samples"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        
        # Save agent state
        xyz = np.concatenate([i.cpu().numpy()[:, :, np.newaxis] for i in [agent_state.pos_x, agent_state.pos_y, agent_state.pos_z]], axis=-1)
        np.save(os.path.join(save_folder, f"agent_state_xyz_{0}.npy"), xyz)
        rotation_as_quaternion = agent_state.rotation_as_quaternion.cpu().numpy()
        np.save(os.path.join(save_folder, f"agent_state_rot_quat_{0}.npy"), rotation_as_quaternion)

        # Save global road graph
        # TODO

        # # assert num_worlds == 1
        # # In first step initialize scene
        # x = GPUDriveGenScene()  # TODO move to __init__
        # x.initialize_scene(env=env, add_background=False, add_map=True, add_coord_system=False)

        # # In subsequent steps just update actor positions
        # x.update_scene(env=env, timestamp=float(t))
        # imgs = x.render_scene(do_save_images=True, timestamp=t)  # (num_observers, num_cameras, H, W, 3)
        # return imgs