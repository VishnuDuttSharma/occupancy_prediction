In [1]: import ai2thor.controller

In [2]: import random

In [3]: local_executable_path = '/home/vishnuds/ai2thor/unity/builds/thor-Linux6
   ...: 4-local/thor-Linux64-local'

In [4]: controller = ai2thor.controller.Controller(local_executable_path=local_e
   ...: xecutable_path,
   ...:            agentMode="bot",
   ...:         #    visibilityDistance=1.5,
   ...:            visibilityDistance=100,
   ...:            scene="FloorPlan201",
   ...:            gridSize=0.10,
   ...:            movementGaussianSigma=0.005,
   ...:            rotateStepDegrees=30,
   ...:            rotateGaussianSigma=0.0,
   ...:            renderDepthImage=True,
   ...:            renderInstanceSegmentation=True,
   ...:            width=640,
   ...:            height=480,
   ...:            fieldOfView=60, ## not correct ,
   ...:            horizon=0
   ...:            )
/home/vishnuds/ai2thor/ai2thor/controller.py:680: UserWarning: On reset and upon initialization, agentMode='bot' has been renamed to agentMode='locobot'.
  warnings.warn(

In [5]: positions = controller.step(
   ...:             action="GetReachablePositions"
   ...:         ).metadata["actionReturn"]

In [6]: position = random.choice(positions)

In [7]: event = controller.step(
   ...:             action="Teleport",
   ...:             position=position
   ...:         )

In [8]: event.metadata['agent']
Out[8]: 
{'name': 'agent',
 'position': {'x': -1.3999998569488525,
  'y': 0.9026570916175842,
  'z': 2.200000047683716},
 'rotation': {'x': -0.0, 'y': 0.0, 'z': 0.0},
 'cameraHorizon': -0.0,
 'projectionMatrix': [[1.299038052558899, 0.0, 0.0, 0.0],
  [0.0, 1.7320507764816284, 0.0, 0.0],
  [0.0, 0.0, -1.0100502967834473, -0.2010050266981125],
  [0.0, 0.0, -1.0, 0.0]],
 'projectionMatrix_inverse': [[0.7698003649711609, 0.0, 0.0, 0.0],
  [0.0, 0.5773502588272095, 0.0, 0.0],
  [-0.0, -0.0, -0.0, -1.0],
  [0.0, 0.0, -4.974999904632568, 5.025000095367432]],
 'cameraToWorldMatrix': [[1.0, 0.0, 0.0, -1.3999998569488525],
  [0.0, 1.0, 0.0, 0.8714570999145508],
  [-0.0, -0.0, -1.0, 2.200000047683716],
  [0.0, 0.0, 0.0, 1.0]],
 'cameraPosition': {'x': -1.3999998569488525,
  'y': 0.8714570999145508,
  'z': 2.200000047683716},
 'cameraEuler': {'x': -0.0, 'y': 0.0, 'z': 0.0},
 'isStanding': None,
 'inHighFrictionArea': False}

In [9]: position = random.choice(positions)

In [10]: event = controller.step(
    ...:             action="Teleport",
    ...:             position=position, rotation={'x':0, 'y':90, 'z':0}
    ...:         )

In [11]: event.metadata['agent']
Out[11]: 
{'name': 'agent',
 'position': {'x': -3.9000000953674316,
  'y': 0.9026570916175842,
  'z': 0.8999999761581421},
 'rotation': {'x': -0.0, 'y': 90.0, 'z': 0.0},
 'cameraHorizon': -0.0,
 'projectionMatrix': [[1.299038052558899, 0.0, 0.0, 0.0],
  [0.0, 1.7320507764816284, 0.0, 0.0],
  [0.0, 0.0, -1.0100502967834473, -0.2010050266981125],
  [0.0, 0.0, -1.0, 0.0]],
 'projectionMatrix_inverse': [[0.7698003649711609, 0.0, 0.0, 0.0],
  [0.0, 0.5773502588272095, 0.0, 0.0],
  [-0.0, -0.0, -0.0, -1.0],
  [0.0, 0.0, -4.974999904632568, 5.025000095367432]],
 'cameraToWorldMatrix': [[-1.1920926112907182e-07,
   -0.0,
   -0.9999998807907104,
   -3.9000000953674316],
  [0.0, 1.0, 0.0, 0.8714570999145508],
  [-0.9999998807907104, -0.0, 1.1920926112907182e-07, 0.8999999761581421],
  [0.0, 0.0, 0.0, 1.0]],
 'cameraPosition': {'x': -3.9000000953674316,
  'y': 0.8714570999145508,
  'z': 0.8999999761581421},
 'cameraEuler': {'x': -0.0, 'y': 90.0, 'z': -0.0},
 'isStanding': None,
 'inHighFrictionArea': False}

In [12]: 
