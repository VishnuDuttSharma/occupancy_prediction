__version__  = '0.1'
__author__ = 'Vishnu Dutt Sharma'
___email__ = 'vishnuds@umd.edu'

# import sys
# sys.path.append('/home/vishnuds/ai2thor')

using_notebook = False

# if using_notebook:
#     !pip install --upgrade ai2thor ai2thor-colab &> /dev/null

import ai2thor
from ai2thor.controller import Controller

import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math 
import copy

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

# print("AI2-THOR Version: " + ai2thor.__version__)


projectionMatrix = np.asarray([[1.299038052558899, 0.0, 0.0, 0.0],
                                [0.0, 1.7320507764816284, 0.0, 0.0],
                                [0.0, 0.0, -1.0100502967834473, -0.2010050266981125],
                                [0.0, 0.0, -1.0, 0.0]]
                             )

projectionMatrix_inverse = np.asarray([[0.7698003649711609, 0.0, 0.0, 0.0],
                                      [0.0, 0.5773502588272095, 0.0, 0.0],
                                      [-0.0, -0.0, -0.0, -1.0],
                                      [0.0, 0.0, -4.974999904632568, 5.025000095367432]]
                                     )

cameraToWorldMatrix = np.asarray([[1.0, 0.0, 0.0, -2.299999952316284],
                                  [0.0, 1.0, 0.0, 0.8714570999145508],
                                  [-0.0, -0.0, -1.0, 2.700000047683716],
                                  [0.0, 0.0, 0.0, 1.0]]
                                )


def d2r(deg):
    """Function to convert angle ion degrees to radians

    Parameters
    ----------
        deg : Angle in degrees

    Returns
    -------
        float: Angle in radians
    """
    return np.pi*deg/180.


#### 
class BotController:
    """Class for the AI2THOR ground robot 
    """
    def __init__(self, init_scene="FloorPlan201", get_depth=True, get_segment=True, fov=90):
        """Constructor for Bot controller. Results in a ai2thor.controller.Controller object. This bot is inteneded to have 4 third party cameras
        in addition to the defaylt camera. The first one is a orthographic, overhead camera looking at the robot from the top. The other cameras 
        are at a different height at the same location (center camera), slightly away towards left and right of the robot. the cameras on the siddes can also have 
        some yaw (user-input).
        
        Parameters
        ----------
            init_scene: Initialization scene. The default is 'FloorPlan201'
            get_depth: Whether to return depth images or not. Default is 'True' i.e. returns the depth images
            get_segment: Whether to return segment images or not. Default is 'True' i.e. returns the depth images
            tp_height: Height for the third party cameras (center, left, and right). Default is 0.5m, same as TurtleBot2
            tp_side_shift: distance of the left and right cameras from the robot. Default is 0.3m
            tp_side_rot: Yaw for the left and right camera, looking towards the front of the robot
            fov: Field of view of the robot-mounted, center, left, and right cameras
            
        Returns
        -------
            None
        """
        
        ## Controller
        self.controller = Controller(
           agentMode="locobot",
        #    visibilityDistance=1.5,
        #    visibilityDistance=100,
           scene=init_scene,
           gridSize=0.10,
           movementGaussianSigma=0.0, #0.005,
           rotateStepDegrees=30,
           rotateGaussianSigma=0.0,
           renderDepthImage=get_depth,
           renderInstanceSegmentation=get_segment,
           width=512, #640,
           height=512,
           fieldOfView=fov, ## not sure,
           horizon=30 ## Robot-mounted camera should not have any pitch
           )
        
        ## Parameters for the side cameras
        self.fov = fov
        self.intrinsic = None
        
        ## Dictionary to save IDs for the third-camera parties
        self.extra_cameras = {}
        self.extrinsic_mat = {}
        
        ## Calculate the intrinsic camers
        self.get_intrinsic_matric()

    def reset(self, scene="FloorPlan201"):
        """Fucntion to reset the robot with a new scene
        
        Parameters
        ----------
            scene: Name of the scene. The default is FloorPlan201, which is the first living room scene
        
        Returns
        -------
            None
        """
        self.controller.reset(scene=scene)

    def get_intrinsic_matric(self):
        """Function to calculate the intrinsic matrix for the robot-mounted (ans the center, left, and right) cameras
        
        Parameters
        ----------
            None
        
        Returns
        -------
            open3d intrindic matrix object: A 3x3 array  intrinsic matrix object
        """
        # Get the width and height of the camera images
        self.width = self.controller.last_event.metadata['screenWidth']
        self.height = self.controller.last_event.metadata['screenHeight']
        
        # Get the Field-of-View of the camera
        self.fov = self.controller.last_event.metadata['fov']
        
        # Helper function to convert degrees to radian
        def to_rad(th):
            return th*math.pi / 180

        # Convert the FoVs to focal lengths
        focal_length_x = 0.5 * self.width * math.tan(to_rad(self.fov/2))
        focal_length_y = 0.5 * self.height * math.tan(to_rad(self.fov/2))

        # Collect the parameters for the intrinsic matrix (focal length for x-axis, focal length for y-axis, center for x-axis, center for y-axis)
        fx, fy, cx, cy = (focal_length_x, focal_length_y, self.width/2, self.height/2)
        
        # Get the inmtrinsic matrix assuming Pinhole Camera using Open3D
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, fx, fy, cx, cy)
        
        # Return the intrinsic matrix
        return self.intrinsic

    def add_tp_cameras(self):
        """Function to add the ground cameras (Center, Left, and Right). The cameras are located at the same height (user-defined).
        Center camera is at the same location as the robot, but different height (user-defined).
        Left camera is at some distance away (lateral, user-defined) towards the left of the robot, and at the same height as the center camera.
        Right camera is at some distance away (lateral, user-defined) towards the right of the robot, and at the same height as the center camera.
        Both the left and right camera can have a yaw (user-defined) looking towards the robot
        
        Parameters
        ----------
            None
            
        Returns
        -------
            None
        """
        # Get the robot position and rotation
        robot_pos = self.controller.last_event.metadata['agent']['position'].copy()
        robot_rot = self.controller.last_event.metadata['agent']['rotation'].copy()
        
        ##### Center Camera Setup
        ## Copy the robot position and rotation
        self.cam_low_tilt = 30
        cam_low_pos = robot_pos.copy()
        cam_low_rot = robot_rot.copy()
        # Set the camera height
        cam_low_pos['y'] = 1.5
        # Set camera rotation
        cam_low_rot['x'] = self.cam_low_tilt
        # Place the third party camera
        event = self.controller.step( 
            action="AddThirdPartyCamera", 
            fieldOfView=self.fov,
            position=cam_low_pos,
            rotation=cam_low_rot
        ) 
        # Add the camera ID to the dictionary
        self.extra_cameras['Low'] = len(self.extra_cameras)
        # getting extrinsic matrix
        low_tilt = d2r(self.cam_low_tilt)
        self.extrinsic_mat['Low'] = np.eye(4)
        self.extrinsic_mat['Low'][1,1] =  np.cos(low_tilt)
        self.extrinsic_mat['Low'][1,2] = -np.sin(low_tilt)
        self.extrinsic_mat['Low'][2,1] =  np.sin(low_tilt)
        self.extrinsic_mat['Low'][2,2] =  np.cos(low_tilt)

        ##### Top Camera Setup
        ## Copy the robot position and rotation
        self.cam_top_tilt = 38
        cam_top_pos = robot_pos.copy()
        cam_top_rot = robot_rot.copy()
        # Set the camera height
        cam_top_pos['y'] = 2.0
        # Set camera rotation
        cam_top_rot['x'] = self.cam_top_tilt
        # Place the third-party camera
        event = self.controller.step( 
            action="AddThirdPartyCamera", 
            fieldOfView=self.fov,
            position=cam_top_pos, 
            rotation=cam_top_rot
        )
        # Add the camera ID to the dictionary
        self.extra_cameras['High'] = len(self.extra_cameras)
        # getting extrinsic matrix
        top_tilt = d2r(self.cam_top_tilt)
        self.extrinsic_mat['High'] = np.eye(4)
        self.extrinsic_mat['High'][1,1] =  np.cos(top_tilt)
        self.extrinsic_mat['High'][1,2] = -np.sin(top_tilt)
        self.extrinsic_mat['High'][2,1] =  np.sin(top_tilt)
        self.extrinsic_mat['High'][2,2] =  np.cos(top_tilt)

    def get_object_table(self):
        """Function to get the object details (position, rotation, object name, etc.). Return these details in a pandas dataframe
        
        Parameters
        ----------
            None
        
        Returns
        -------
            pandas.Dataframe: Dataframe containing the details of the object in the scene
        """
        return pd.DataFrame(self.controller.last_event.metadata['objects'])


    def move_object_by_diff(self, object_type="Dining Table", x=0, y=3.0, z=0):
        """Function to displace an object in the scene.
        
        Parameters
        ----------
            object_type: Type of the object. Deafult is 'Dininig Table'.
            x: Displacement in the x-axis
            y: Displacement in the y-axis (perpendicular to the ground)
            z: Displacement in the z-axis
            
        Returns
        -------
            dict: Dictionary containing the updated position and rotation of each object
        """
        # Get the object details/table
        data_table = get_object_table()
        # Find the desired object by type in the table (assuming there is only one object of this type in the scene)
        index = data_table[data_table.objectType == object_type].index[0]
        
        ### We need to copy pos of all the objects as the object whose pose is not set when calling the API gets removed from the scene
        # Get the position and rotation of all the objects in the scene
        all_objects = self.controller.last_event.metadata['objects'].copy()
        # Create a place holder for copying these objects to remember the original location
        all_objects_copy = []
        # Copy the pose for all the objects
        for obj in all_objects:
            all_objects_copy.append({
                'objectName': obj['name'],
                'rotation': obj['rotation'],
                'position': obj['position']
            })
        
        # Find the previous position of the desired object
        prev_position = all_objects_copy[index]['position'].copy()
        # Displace the object and update in the new position list
        new_position = prev_position.copy()
        new_position['x'] = new_position['x'] + x
        new_position['y'] = new_position['y'] + y
        new_position['z'] = new_position['z'] + z
        all_objects_copy[index]['position'] = new_position
        
        # Update the object position
        self.controller.step(action='SetObjectPoses',
                        objectPoses=all_objects_copy)
        
        # Return the updated position list
        return all_objects_copy

    def move_object_by_obj(self, all_object_pos, is_refined=True):
        """Function to move objects using disctionary of pose
        
        Parameters
        ----------
            all_object_pos: Dictionary containing the objects' pos (position and rotation). The keyas are the names of the objects
            is_refined: Indicates whetehr the data has been processed or is taken from the API (and needs refining)
        
        Returns
        -------
            None
        """
        ## if the disctionary is not refined (it has elements other than the pose), it will be refined
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
        
        # update the objects' pose
        self.controller.step(action='SetObjectPoses',
                        objectPoses=new_object_pos)
    
    def update_tp_cameras(self):
        """Function to update the poses of the third party camera (center, left, and right) based on the robot's pose
        
        Parameters
        ----------
            None
        
        Returns
        -------
            None
        """
        # Get the robot's pose
        robot_pos = self.controller.last_event.metadata['agent']['position'].copy()
        robot_rot = self.controller.last_event.metadata['agent']['rotation'].copy()
        

        ##### Center Camera
        ## Copy the robot position and rotation
        cam_low_pos = robot_pos.copy()
        cam_low_rot = robot_rot.copy()
        # Set the camera height
        cam_low_pos['y'] = 1.5
        # Set camera rotation
        cam_low_rot['x'] = self.cam_low_tilt
        # Place the third party camera
        # Update the third-party camers pose
        event = self.controller.step( 
            action="UpdateThirdPartyCamera",
            thirdPartyCameraId=self.extra_cameras['Low'],
            position=cam_low_pos,
            rotation=cam_low_rot,
            fieldOfView=self.fov
        )
        
        # Copy the robot position
        cam_top_pos = robot_pos.copy()
        cam_top_rot = robot_rot.copy()
        # Set the camera height
        cam_top_pos['y'] = 2.0
        # Set camera rotation
        cam_top_rot['x'] = self.cam_top_tilt
        # Update the third-part camera pose
        event = self.controller.step( 
            action="UpdateThirdPartyCamera",
            thirdPartyCameraId=self.extra_cameras['High'],
            position=cam_top_pos, 
            rotation=cam_top_rot,
            fieldOfView=self.fov
        ) 
        
    
    def get_reacheble_pos(self):
        """Function to get the reachable positions on the map for the robot
        
        Parameters
        ----------
            None
        
        Returns
        -------
            list: List of positions (dictionary with keys 'x', 'y', and 'z') that are reachble for the robot on the map/scene
        """
        # Get the reachable positions
        positions = self.controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]
        
        # Return the positions
        return positions

    def goto_random_pos(self, positions=None):
        """Function to move the robot to a random position out of the candidate reachable positions
        
        Parameters
        ----------
            positions: A list of positions (dictionaries with the keys 'x', 'y', and 'z'). If set to 'None' (default) a random location
                        out of all the reachable positions is selected to move to
            
        Returns
        -------
            None
        """
        # If no list of positions is given, then all reachable positions are used as teh candidates
        if positions is None:
            positions = self.get_reacheble_pos()
        
        # Select a random position
        position = random.choice(positions)
        
        # Move the robot to teh select position
        self.controller.step(
            action="Teleport",
            position=position
        )
        
        # Update the location of the third party cameras (center, left, and right cameras)
        self.update_tp_cameras()

    def get_images(self, scene=True, depth=True, segment=True):
        """Function to get the images from the third party camras (center, left, and right cameras)
        
        Parameters
        ----------
            scene: Whether to return the scene images. Default is True
            depth: Whether to return the depth images. Default is True
            segment: Whether to return the segment images. Default is True
        
        Returns
        -------
            dict: Dictionary containing the images from the third party cameras as a dictionary
        """
        # Create a placeholder for images
        img_dict = {}
        
        # Add scene images, if set to true
        if scene:
            scene_images = {}
            scene_images['Low'] = self.controller.last_event.third_party_camera_frames[self.extra_cameras['Low']]
            scene_images['High'] = self.controller.last_event.third_party_camera_frames[self.extra_cameras['High']]
            img_dict['scene'] = scene_images
        
        # Add depth images, if set to true
        if depth:
            depth_images = {}
            depth_images['Low'] = self.controller.last_event.third_party_depth_frames[self.extra_cameras['Low']]
            depth_images['High'] = self.controller.last_event.third_party_depth_frames[self.extra_cameras['High']]
            img_dict['depth'] = depth_images
        
        # Add segemnt images, if set to true
        if segment:
            seg_images = {}
            seg_images['Low'] = self.controller.last_event.third_party_instance_segmentation_frames[self.extra_cameras['Low']]
            seg_images['High'] = self.controller.last_event.third_party_instance_segmentation_frames[self.extra_cameras['High']]
            img_dict['segment'] = seg_images
        
        return img_dict
    
    
    
    def get_point_cloud(self, color_img, depth_img, extrinsic_mat):
        """Function to get the point clound using the depth and segmentation images
        
        Parameters
        ----------
            color_img: Segmentation or scene image
            depth_img: Depth image
            
        Returns
        -------
            open3d.pointcloud: PointCloud object from open3d 
        """
        # Convert the color image to an open3d image
        color = o3d.geometry.Image(color_img.astype(np.uint8))
        
        # Convert the depth image to an open3d image
        depth = o3d.geometry.Image(depth_img)
        
        # Get rgbd image form the color and depth images
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth,
                                                            depth_scale=1.0,
                                                            # depth_trunc=0.7,
                                                            depth_trunc=10.0,
                                                            convert_rgb_to_intensity=False)
        # Get the poin cloud using camera instrinsic matrix 
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic, extrinsic=extrinsic_mat)
        
        ### uncomment to change the frame
        # pcd.transform([[1, 0, 0, 0],
        #         [0, -1, 0, 0],
        #         [0, 0, -1, 0],
        #         [0, 0, 0, 1]])

        return pcd
    
    def point_filter(self, pcd, exclusion_list):
        """Function to return filter based on colors
        
        Parameters
        ----------
            pcd: Point cloud which is required to be filtered. It should have points and colors attributes
            exlusion_list: List of colors (normalized) which are to be exceluded from the point cloud
        
        Returns
        -------
            np.ndarray: 1-D array of booleans indicating which points should be kept to exclude the colors passed as input argument
        """
        # Get the points and colors from teh point cloud and convert them to numpy arrays (otherwise they are open3d objects)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Create a placeholder for teh filter
        filter_idx = np.zeros((len(colors),), dtype=np.bool)
        
        # Iterate over the colors in the exclusion list and set filter to true if a match is found. += in this fucntion simulated bitwise OR
        for col in exclusion_list:
            filter_idx += (colors == np.array(col)/255.).all(axis=1)
        
        # Return the inversion of the filter as we want to remove the points given in the exclusion list
        return ~filter_idx #pcd.select_by_index(filter[:,0])

    def get_all_point_clouds(self):
        """Function to get the point clouds from all the ground cameras (center, left, right)
        
        Parameters
        ----------
            None
        
        Returns
        -------
            list: List of open3d point clouds from center, left, and right cameras respectively
        """
        img_dict = self.get_images()
        pcd_list = []
        ## Low cam
        scene_img = img_dict['segment']['Low']
        depth_img = img_dict['depth']['Low']

        low_pcd = self.get_point_cloud(scene_img, depth_img, self.extrinsic_mat['Low'])
        pcd_list.append(copy.copy(low_pcd))

        ## High cam
        scene_img = img_dict['segment']['High']
        depth_img = img_dict['depth']['High']

        high_pcd = self.get_point_cloud(scene_img, depth_img, self.extrinsic_mat['High'])
        pcd_list.append(copy.copy(high_pcd))

        return pcd_list


def point_filter(pcd, exclusion_list):
        """Function to return filter based on colors
        
        Parameters
        ----------
            pcd: Point cloud which is required to be filtered. It should have points and colors attributes
            exlusion_list: List of colors (normalized) which are to be exceluded from the point cloud
        
        Returns
        -------
            np.ndarray: 1-D array of booleans indicating which points should be kept to exclude the colors passed as input argument
        """
        # Get the points and colors from teh point cloud and convert them to numpy arrays (otherwise they are open3d objects)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Create a placeholder for teh filter
        filter_idx = np.zeros((len(colors),), dtype=np.bool)
        
        # Iterate over the colors in the exclusion list and set filter to true if a match is found. += in this fucntion simulated bitwise OR
        for col in exclusion_list:
            filter_idx += (colors == np.array(col)/255.).all(axis=1)
        
        # Return the inversion of the filter as we want to remove the points given in the exclusion list
        return ~filter_idx #pcd.select_by_index(filter[:,0])

def pcd_to_raw_map(pcd, xlims=(-2.5,2.5), ylims=(0,5), map_size=(256,256)):
    """
    Function to convert point cloud to raw 2D occupany map/Histogram.
    The resulting histogram needs to transposed and vertically flipped to match with teh world frame image

    Parameters
    ----------
        pcd: Point cloud, an open3d.Pointcloud object
        xlims: Limits of the x-coordinates of point cloud as tuple
        ylims: Limits of the y-coordinates of point cloud as tuple
        map_size: Size of target map/image size as tuple

    Returns
    -------
        np.ndarray: Histogram as a 2D array of dimentios defined by the map_size parameter
    """
    # Get points from Point cloud
    points = np.asarray(pcd.points)

    # Keep only the point that are within the limit
    filter_ids = (points[:,0] >= xlims[0]) & (points[:,0] <= xlims[1]) & (points[:,2] >= ylims[0]) & (points[:,2] <= ylims[1])
    # Filter data
    pcd = pcd.select_by_index(np.where(filter_ids == True)[0])
    
    # Getting points from the filtered point cloud
    points = np.asarray(pcd.points)
    
    '''Legacy debugger
    colors = pcd.colors

    # points[:,2] = ylims[0] - points[:,2]
    # return points, colors
    '''

    # Rescale and quantize the points (Minmax scaling -> rescaling -> quantization by rounding)
    x_pts = np.round((map_size[0]-1) * (points[:,0] - xlims[0])/(xlims[1] - xlims[0])).astype(int)
    y_pts = np.round((map_size[1]-1) * (points[:,2] - ylims[0])/(ylims[1] - ylims[0])).astype(int)
    
    # Creating the map (as a histogram)
    raw_map = np.zeros(map_size, dtype=int)
    for i_pt in range(len(x_pts)):
        raw_map[x_pts[i_pt], y_pts[i_pt]] += 1

    return raw_map


def get_occ_map(pcd, floor_colors, ceiling_colors, ll_factor=0.01, max_pts=10):
    """Function to get the occupancy map
    """
    filter_ido = point_filter(pcd, floor_colors+ceiling_colors)
    occ_pcd = pcd.select_by_index(np.where(filter_ido == True)[0])
    occ_occ_map = pcd_to_raw_map(occ_pcd)
    
    filter_idf = point_filter(pcd, floor_colors)
    free_pcd = pcd.select_by_index(np.where(filter_idf == False)[0])
    free_occ_map = pcd_to_raw_map(free_pcd)

    comb_occ_map = occ_occ_map - free_occ_map

    occ_map =  np.clip(comb_occ_map, a_min=-max_pts, a_max=max_pts).astype(np.uint8)

    return occ_map

