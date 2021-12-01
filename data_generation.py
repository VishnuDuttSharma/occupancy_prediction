#import sys
#sys.path.append('/home/vishnuds/ai2thor')

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

from helper import *

from tqdm import tqdm


g_count = 0
inp_dir = 'inp_data/'
gt_dir = 'gt_data/'
descfile = 'description_ang0.csv'

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

    controller.add_top_camera()
    controller.add_side_cameras()

    positions = controller.get_reacheble_pos()
    print(f'Reachable positions: {len(positions)}')
    floor_color = [x['color'] for x in controller.controller.last_event.metadata['colors'] if 'floor' in x['name'].lower()]
    ceiling_color = [x['color'] for x in controller.controller.last_event.metadata['colors'] if 'ceiling' in x['name'].lower()]

    for pos in  positions:
        for yaw in range(0,360,45):#range(0,360,30):
            orient = {'x': 0, 'y':yaw, 'z': 0}

            event = controller.controller.step(
                        action="Teleport",
                        position=pos,
                        rotation=orient
                    )

            controller.update_side_cameras()
            pc_list = controller.get_all_point_clouds()

            main_pcd = pc_list[0]
            inp_occ_map = occ_map = get_occ_map(main_pcd, floor_color, ceiling_color)

            main_pcd = pc_list[1]
            left_occ_map = occ_map = get_occ_map(main_pcd, floor_color, ceiling_color)

            main_pcd = pc_list[2]
            right_occ_map = occ_map = get_occ_map(main_pcd, floor_color, ceiling_color)

            gt_map = np.maximum(np.abs(inp_occ_map), np.maximum(np.abs(left_occ_map), np.abs(right_occ_map))) * np.sign(inp_occ_map + left_occ_map + right_occ_map)

            free_perc = 100*np.sum(gt_map == 0)/gt_map.size

            file_name = f'FP{fl_num}_{g_count}'
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

            np.save(inp_dir+file_name+'.npy', arr=inp_occ_map)
            np.save(gt_dir+file_name+'.npy', arr=gt_map)

            df = df.append(desc, ignore_index=True)

            g_count += 1

df.to_csv(descfile)


