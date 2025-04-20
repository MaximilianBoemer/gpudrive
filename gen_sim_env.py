
import os

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

from gpudrive.datatypes.observation import GlobalEgoState
from gpudrive.datatypes.roadgraph import GlobalRoadGraphPoints
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.config import RenderConfig

from perception_simulation.asset_utils import PoseData
from perception_simulation.perception_sim import Scene, MapScene, DynamicScene
from perception_simulation.scene_config import SceneConfig, DynamicSceneConfig, EgoConfig, StaticSceneConfig, RelightingConfig


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
        self.add_image_obs = True

        # TODO - move out of here and have it as an arg
        self.scene_config = SceneConfig(
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
                ego_agent_id=0  # TODO - redesign to have option to have multiple ego agents - maybe get rid of ego agent concept at all
            ),
            gaussian_type = "2D",
            save_folder="/n/fs/pci-sharedt/mb9385/workspace/gpudrive/save"
        )
        relighting_config = RelightingConfig(
            method=None,
            params={}
        )
        self.gen_sim_scene = Scene(
            scene_config=self.scene_config,
            relighting_config=relighting_config
        )
        self.initialize_scene(
            add_background=False,
            add_map=False,
            add_coord_system=True
        )
        print("Generated scene initialized")

    def initialize_scene(
            self,
            add_background: bool = True,
            add_map: bool = False,
            add_coord_system: bool = False
        ):
        
        global_roadgraph = GlobalRoadGraphPoints.from_tensor(
            roadgraph_tensor=self.sim.map_observation_tensor(),
            backend=self.backend,
            device=self.device,
        )

        # TODO - Enable Map
        map_scene = None  # MapScene(gaussian_type=self.scene_config.gaussian_type)

        dynamic_scene = DynamicScene(gaussian_type=self.scene_config.gaussian_type)
        # TODO - Enable loading asset library in the beginning and do it very fast
        world_id = 0
        for agent_id in range(self.max_cont_agents):
            if self.cont_agent_mask[world_id, agent_id].item() is False:
                continue
            dynamic_scene.add_new_agent(
                agent_id=agent_id,
                car_type="blue_panda",  # TODO come up with some pseudo random initialization scheme
                pose_data=PoseData()
            )

        self.gen_sim_scene.initialize_scene(
            map_scene=map_scene,
            dynamic_scene=dynamic_scene,
            add_background=add_background,
            add_map=add_map,
            add_coord_system=add_coord_system
        )

    def get_obs(self, mask=None, time_step=None):
        """Get observation: Combine different types of environment information into a single tensor.

        Returns:
            torch.Tensor: (num_worlds, max_agent_count, num_features)
        """
        ego_states = self._get_ego_state(mask)
        partner_observations = self._get_partner_obs(mask)
        road_map_observations = self._get_road_map_obs(mask)
        lidar_observations = self._get_lidar_obs(mask)
        if getattr(self, 'add_image_obs', False):
            image_observations = self._get_image_obs(mask)
            # TODO - Add image observations to the obs
            obs = torch.cat(
                (
                    ego_states,
                    partner_observations,
                    road_map_observations,
                ),
                dim=-1,
            )
            save = False
            if save:
                save_images(
                    imgs=image_observations,
                    file_name=f"{time_step}.png",
                    save_folder="/n/fs/pci-sharedt/mb9385/workspace/gpudrive/sample_images"
                )

        else:
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

        world_id = 0
        agent_state = GlobalEgoState.from_tensor(
            self.sim.absolute_self_observation_tensor(),
            self.backend,
            device=self.device,
        )

        # TODO - This should be at initialization stage
        global_roadgraph = GlobalRoadGraphPoints.from_tensor(
            roadgraph_tensor=self.sim.map_observation_tensor(),
            backend=self.backend,
            device=self.device,
        )

        # update scene
        present_agents = torch.where(self.cont_agent_mask[0])[0].cpu().numpy()
        assert all(sorted([a.agent_id for a in self.gen_sim_scene.dynamic_scene.agents] + [self.gen_sim_scene.ego_agent.agent_id]) == present_agents)
        # TODO - add new other agents
        # TODO - remove all actors that are not present anymore

        # Get + update poses
        # poses = self.gen_sim_scene.get_object_poses(present_objects=present_objects, timestamp=timestamp)
        # TODO rewrite this
        self.gen_sim_scene.update_scene(
            poses={
                agent_id: PoseData(
                    translation=np.concatenate([i.cpu().numpy()[:, :, np.newaxis] for i in [agent_state.pos_x, agent_state.pos_y, agent_state.pos_z]], axis=-1)[world_id][agent_id],
                    rotation_quat=agent_state.rotation_as_quaternion.cpu().numpy()[world_id][agent_id]
                ) for agent_id in present_agents
            }
        )
        print(self.gen_sim_scene.ego_agent.pose_data.translation)
        
        # TODO - In Multi agent case one needs to support rendering for all agents and just add an additional ego mask
        # Only render when still active
        imgs = self.gen_sim_scene.render_scene()
        return imgs


def create_joint_img(images: dict, scale: float = 1.0, variant: str = "default"):
    third_person = images["THIRD_PERSON"]
    img_width = images["FRONT_LEFT"].shape[2]

    if variant == "default":
        scale_factor = (img_width * 3) / third_person.shape[2]
        joint_img_top = torch.nn.functional.interpolate(
            third_person.unsqueeze(0), 
            scale_factor=(scale_factor, 3), 
            mode="bilinear", 
            align_corners=False
        ).squeeze(0)
        
        joint_img_bottom = torch.cat([
            torch.cat([images["FRONT_LEFT"], images["FRONT"], images["FRONT_RIGHT"]], dim=2),
            torch.cat([images["SIDE_LEFT"], torch.zeros_like(images["SIDE_LEFT"]), images["SIDE_RIGHT"]], dim=2)
        ], dim=1)

        joint_img = torch.cat([joint_img_top, joint_img_bottom], dim=1)
    elif variant == "iccv25":
        joint_img = torch.cat([images["THIRD_PERSON"], images["FRONT_LEFT"], images["FRONT"], images["FRONT_RIGHT"]], dim=2)
    else:
        raise NotImplementedError()
    if scale != 1.0:
        joint_img = torch.nn.functional.interpolate(
            joint_img.unsqueeze(0), 
            scale_factor=scale, 
            mode="bilinear", 
            align_corners=False
        ).squeeze(0)
    return joint_img


def save_images(
        imgs: dict,
        file_name: str,
        save_folder: str,
        save_individual_frames: bool = True,
        save_joint_img: bool = True,
        scale_joint_img: float = 0.125,
        joint_img_variant: str = "default"
    ):
    os.makedirs(save_folder, exist_ok=True)
    
    if save_individual_frames:
        for cam_name, img in imgs.items():
            if cam_name not in ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "THIRD_PERSON"]:
                continue
            cam_save_folder = os.path.join(save_folder, cam_name)
            os.makedirs(cam_save_folder, exist_ok=True)
            if save_individual_frames:
                to_pil_image(img).save(os.path.join(cam_save_folder, file_name))
    
    if save_joint_img:
        joint_img = create_joint_img(images=imgs, scale=scale_joint_img, variant=joint_img_variant)
        os.makedirs(os.path.join(save_folder, "joint"), exist_ok=True)
        pil_image = to_pil_image(joint_img)
        pil_image.save(os.path.join(save_folder, "joint", file_name))
