#import sys
#sys.path.append('/home/vishnuds/ai2thor')

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

from baseline_helper import *

from tqdm import tqdm

np.random.seed(1234)

g_count = 0
inp_dir = './baseline_data/inp_data/'
gt_dir = './baseline_data/gt_data/'
descfile = './baseline_data/description_ang0.csv'

df = pd.DataFrame(columns=['Filename',
                           'FloorName',
                           'pos_x',
                           'pos_y',
                           'pos_z',
                           'ang_x',
                           'ang_y',
                           'ang_z',
                           'free_perc'])

for fl_num in tqdm(range(201, 231)):
    controller = BotController(init_scene=f'FloorPlan{fl_num}')

    controller.add_tp_cameras()

    positions = controller.get_reacheble_pos()
    print(f'Reachable positions: {len(positions)}')
    floor_color = [x['color'] for x in controller.controller.last_event.metadata['colors'] if 'floor' in x['name'].lower()]
    ceiling_color = [x['color'] for x in controller.controller.last_event.metadata['colors'] if 'ceiling' in x['name'].lower()]

    
    for cnt in range(500):
        pos = np.random.choice(positions)
        yaw = np.random.randint(low=0, high=359)
        orient = {'x': 0, 'y':yaw, 'z': 0}

        event = controller.controller.step(
                    action="Teleport",
                    position=pos,
                    rotation=orient
                )

        controller.update_tp_cameras()
        pc_list = controller.get_all_point_clouds()

        main_pcd = pc_list[0]
        low_occ_map = get_occ_map(main_pcd, floor_color, ceiling_color)

        main_pcd = pc_list[1]
        high_occ_map = get_occ_map(main_pcd, floor_color, ceiling_color)

        inp_map = low_occ_map
        gt_map = np.maximum(np.abs(low_occ_map), np.abs(high_occ_map)) * np.sign(low_occ_map + high_occ_map)
        
        free_perc = 100*np.sum(gt_map == 0)/gt_map.size
        occ_perc = 100*np.sum(inp_map != 0)/inp_map.size

        file_name = f'FP{fl_num}_{cnt}'
        desc = {}
        desc['Filename'] = file_name
        desc['FloorName'] = f'FloorPlan{fl_num}'
        desc['pos_x'] = pos['x']
        desc['pos_y'] = pos['y']
        desc['pos_z'] = pos['z']
        desc['ang_x'] = orient['x']
        desc['ang_y'] = orient['y']
        desc['ang_z'] = orient['z']
        desc['free_perc'] = free_perc
        desc['occ_perc'] = occ_perc

        np.save(inp_dir+file_name+'.npy', arr=inp_map)
        np.save(gt_dir+file_name+'.npy', arr=gt_map)

        df = df.append(desc, ignore_index=True)

    

df.to_csv(descfile)


