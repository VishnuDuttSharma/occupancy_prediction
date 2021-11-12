using_notebook = True

# if using_notebook:
#     !pip install --upgrade ai2thor ai2thor-colab &> /dev/null

import ai2thor
from ai2thor.controller import Controller

import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math 
import open3d as o3d

if using_notebook:
    import ai2thor_colab

    from ai2thor_colab import (
        plot_frames,
        show_objects_table,
        side_by_side,
        overlay,
        show_video
    )

    ai2thor_colab.start_xserver()

print("AI2-THOR Version: " + ai2thor.__version__)


projectionMatrix = np.asarray([[1.299038052558899, 0.0, 0.0, 0.0],
                                [0.0, 1.7320507764816284, 0.0, 0.0],
                                [0.0, 0.0, -1.0100502967834473, -0.2010050266981125],
                                [0.0, 0.0, -1.0, 0.0]]
                             )

projectionMatrix_inverse = np.asarray([[1.299038052558899, 0.0, 0.0, 0.0],
                                       [0.0, 1.7320507764816284, 0.0, 0.0],
                                       [0.0, 0.0, -1.0100502967834473, -0.2010050266981125],
                                       [0.0, 0.0, -1.0, 0.0]]
                                     )

cameraToWorldMatrix = np.asarray([[1.0, 0.0, 0.0, -2.299999952316284],
                                  [0.0, 1.0, 0.0, 0.8714570999145508],
                                  [-0.0, -0.0, -1.0, 2.700000047683716],
                                  [0.0, 0.0, 0.0, 1.0]]
                                )






#### 
class BotController:
    def __init__(self, init_scene="FloorPlan201", get_depth=True, get_segment=True, tp_height=0.50, tp_side_shift=0.30, tp_side_rot=30):
        self.controller = Controller(
           agentMode="bot",
        #    visibilityDistance=1.5,
           visibilityDistance=100,
           scene=init_scene,
           gridSize=0.10,
           movementGaussianSigma=0.005,
           rotateStepDegrees=30,
           rotateGaussianSigma=0.0,
           renderDepthImage=get_depth,
           renderInstanceSegmentation=get_segment,
           width=640,
           height=480,
           fieldOfView=60, ## not correct ,
           horizon=0
           )

        self.tp_height = tp_height
        self.tp_side_shift  = tp_side_shift
        self.tp_side_rot = tp_side_rot
        self.extra_cameras = {}

        self.get_intrinsic_matric

    def reset(self, scene="FloorPlan201"):
        self.controller.reset(scene=scene)

    def get_intrinsic_matric(self):
        self.width = self.controller.last_event.metadata['screenWidth']
        self.height = self.controller.last_event.metadata['screenHeight']

        self.fov = self.controller.last_event.metadata['fov']
        def to_rad(th):
            return th*math.pi / 180

        # Convert fov to focal length
        focal_length_x = 0.5 * self.width * math.tan(to_rad(self.fov/2))
        focal_length_y = 0.5 * self.height * math.tan(to_rad(self.fov/2))

        # camera intrinsics
        fx, fy, cx, cy = (focal_length_x, focal_length_y, self.width/2, self.height/2)

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, fx, fy, cx, cy)
        return self.intrinsic

    def add_top_camera(self, height=2.5):
        scene_bounds = self.controller.last_event.metadata['sceneBounds']['center']
        self.controller.step(
            action="AddThirdPartyCamera",
            position=dict(x=scene_bounds['x'], y=height, z=scene_bounds['z']),
            rotation=dict(x=90, y=0, z=0),
            orthographic=True,
            orthographicSize=3.25,
            skyboxColor="white"
            )


        self.extra_cameras['Top'] = len(self.extra_cameras)

    def add_side_cameras(self):
        robot_pos = self.controller.last_event.metadata['agent']['position'].copy()
        robot_rot = self.controller.last_event.metadata['agent']['rotation'].copy()

        cam_center_pos = robot_pos.copy()
        cam_center_rot = robot_rot.copy()
        cam_center_pos['y'] = self.tp_height
        event = self.controller.step( 
            action="AddThirdPartyCamera", 
            fieldOfView=60,
            position=cam_center_pos,
            rotation=cam_center_rot
        ) 
        self.extra_cameras['Center'] = len(self.extra_cameras)

        cam_left_pos = robot_pos.copy()
        cam_left_pos['y'] = self.tp_height
        cam_left_pos['x'] -= self.tp_side_shift
        cam_left_rot = robot_rot.copy()
        cam_left_rot['y'] -= self.tp_side_rot
        event = self.controller.step( 
            action="AddThirdPartyCamera", 
            fieldOfView=60,
            position=cam_left_pos, 
            rotation=cam_left_rot
        ) 
        self.extra_cameras['Right'] = len(self.extra_cameras)

        cam_right_pos = robot_pos.copy()
        cam_right_pos['y'] = self.tp_height
        cam_right_pos['x'] += self.tp_side_shift
        cam_right_rot = robot_rot.copy()
        cam_right_rot['y'] += self.tp_side_rot
        event = self.controller.step( 
            action="AddThirdPartyCamera", 
            fieldOfView=60,
            position=cam_right_pos, 
            rotation=cam_right_rot
        ) 
        self.extra_cameras['Left'] = len(self.extra_cameras)


    def get_object_table(self):
        return pd.DataFrame(self.controller.last_event.metadata['objects'])


    def move_object_by_diff(self, object_Type="Dining Table", x=0, y=3.0, z=0):
        data_table = get_object_table()
        index = data_table[data_table.objectType == object_Type].index[0]

        all_objects = self.controller.last_event.metadata['objects'].copy()
        all_objects_copy = []
        for obj in all_objects:
            all_objects_copy.append({
                'objectName': obj['name'],
                'rotation': obj['rotation'],
                'position': obj['position']
            })

        prev_position = all_objects_copy[index]['position'].copy()
        new_position = prev_position.copy()
        new_position['x'] = new_position['x'] + x
        new_position['y'] = new_position['y'] + y
        new_position['z'] = new_position['z'] + z
        all_objects_copy[index]['position'] = new_position


        self.controller.step(action='SetObjectPoses',
                        objectPoses=all_objects_copy)

        return all_objects_copy

    def move_object_by_obj(self, all_object_pos, is_refined=True):
        if is_refined:
            new_object_pos = all_object_pos.copy()
        else:
            new_object_pos = []
            for obj in all_object_pos:
                new_object_pos.append({
                    'objectName': obj['name'],
                    'rotation': obj['rotation'],
                    'position': obj['position']
                })

        self.controller.step(action='SetObjectPoses',
                        objectPoses=new_object_pos)
    
    def update_side_cameras(self):
        robot_pos = self.controller.last_event.metadata['agent']['position'].copy()
        robot_rot = self.controller.last_event.metadata['agent']['rotation'].copy()

        cam_center_pos = robot_pos.copy()
        cam_center_rot = robot_rot.copy()
        cam_center_pos['y'] = self.tp_height
        event = self.controller.step( 
            action="UpdateThirdPartyCamera",
            thirdPartyCameraId=self.extra_cameras['Center'],
            position=cam_center_pos,
            rotation=cam_center_rot,
            fieldOfView=60
        )

        cam_left_pos = robot_pos.copy()
        cam_left_pos['y'] = self.tp_height
        cam_left_pos['x'] -= self.tp_side_shift
        cam_left_rot = robot_rot.copy()
        cam_left_rot['y'] -= self.tp_side_rot
        event = self.controller.step( 
            action="UpdateThirdPartyCamera",
            thirdPartyCameraId=self.extra_cameras['Left'],
            position=cam_left_pos, 
            rotation=cam_left_rot,
            fieldOfView=60
        ) 

        cam_right_pos = robot_pos.copy()
        cam_right_pos['y'] = self.tp_height
        cam_right_pos['x'] += self.tp_side_shift
        cam_right_rot = robot_rot.copy()
        cam_right_rot['y'] += self.tp_side_rot
        event = self.controller.step( 
            action="UpdateThirdPartyCamera",
            thirdPartyCameraId=self.extra_cameras['Right'],
            position=cam_right_pos, 
            rotation=cam_right_rot,
            fieldOfView=60
        ) 

    def get_reacheble_pos(self):
        positions = self.controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]

        return positions

    def goto_random_pos(self, positions=None):
        if positions is None:
            positions = self.get_reacheble_pos()

        position = random.choice(positions)
        self.controller.step(
            action="Teleport",
            position=position
        )
        self.update_side_cameras()

    def get_images(self, scene=True, depth=True, segment=True):
        img_dict = {}
        if scene:
            scene_images = {}
            scene_images['Center'] = self.controller.last_event.third_party_camera_frames[self.extra_cameras['Center']]
            scene_images['Left'] = self.controller.last_event.third_party_camera_frames[self.extra_cameras['Left']]
            scene_images['Right'] = self.controller.last_event.third_party_camera_frames[self.extra_cameras['Right']]
            img_dict['scene'] = scene_images
        
        if depth:
            depth_images = {}
            depth_images['Center'] = self.controller.last_event.third_party_depth_frames[self.extra_cameras['Center']]
            depth_images['Left'] = self.controller.last_event.third_party_depth_frames[self.extra_cameras['Left']]
            depth_images['Right'] = self.controller.last_event.third_party_depth_frames[self.extra_cameras['Right']]
            img_dict['depth'] = depth_images
        
        if segment:
            seg_images = {}
            seg_images['Center'] = self.controller.last_event.third_party_instance_segmentation_frames[self.extra_cameras['Center']]
            seg_images['Left'] = self.controller.last_event.third_party_instance_segmentation_frames[self.extra_cameras['Left']]
            seg_images['Right'] = self.controller.last_event.third_party_instance_segmentation_frames[self.extra_cameras['Right']]
            img_dict['segment'] = seg_images
        
        return img_dict
    
