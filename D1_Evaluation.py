import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def apply_mask(image, mask, color, alpha=0.4):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  (1-alpha)*image[:, :, c] + alpha * color[c],
                                  image[:, :, c])
    return image


data = pd.read_csv('./data/value_table.csv', index_col=0)

# locate sprinkler at hypothesis
t1 = [10, 30, 50, 70, 90, 110, 130, 149]
t2 = [10, 40, 70, 100, 130]
t3 = [10, 50, 90, 130]
t1 = np.isin(data['height'].values, t1) * 1
t2 = np.isin(data['height'].values, t2) * 1
t3 = np.isin(data['height'].values, t3) * 1
data['t20'] = t1
data['t30'] = t2
data['t40'] = t3

# add my result
result1 = pd.read_csv('./data/result_300_340.csv', index_col=0 )
result2 = pd.read_csv('./data/result_600_640.csv', index_col=0 )
data['r_300_340'] = result1['locate']
data['r_600_640'] = result2['locate']



############################################
# Visualization
env_dict = np.load('./data/env_dict.npy', allow_pickle=True).item()
env_map = env_dict['land'].copy()
env_map2 = np.zeros((env_map.shape[0], env_map.shape[1], 3))
for component in ['house', 'structure', 'tree', 'facility']:
    c_map = env_dict[component][0]
    c_label = env_dict[component][1]
    c_color = env_dict[component][2]
    env_map[1 == c_map] = c_label
    env_map2[1 == c_map, :] = np.array(c_color)/255.

for loc in ['t20', 't30', 't40', 'r_300_340', 'r_600_640']:
    env_vis_map = env_map2.copy()
    result = data[['width', 'height', loc]]
    for row in result.values:
        sprinkler = env_dict['land'].copy()
        w = row[0] * 10  # 10베 축소하였기 다시 확
        h = row[1] * 10
        locate = row[2]
        if w == 1000:
            w = 999
        if locate == 1:
            cv2.circle(sprinkler,
                       (int(h), int(w)),
                       300, 1, -1)
            cv2.circle(env_vis_map,
                       (int(h), int(w)),
                       20, [0, 0, 1], -1)
            apply_mask(env_vis_map, sprinkler, color=[0, 0, 0.1], alpha=0.4)
    plt.imsave(f'./data/result_fig_{loc}.jpg', env_vis_map)


############################################
# eval
for var_name in ['t20', 't30', 't40', 'r_300_340', 'r_600_640']:
    var_area = data.columns.values[3: 3+19]
    dd = data[data[var_name] == 1]
    water = dd[var_area].sum().values
    print(f'{var_name} : count={dd.shape[0]}. mean={np.mean(water)}, std={np.std(water)}')

'''
t20 : count=24. mean=843.7894736842105, std=26.378976771521394
t30 : count=15. mean=540.0, std=73.19260819739137
t40 : count=12. mean=425.42105263157896, std=55.926940866468286
r_300_340 : count=10. mean=320.36842105263156, std=12.444447192542457
r_600_640 : count=18. mean=625.3684210526316, std=13.685222634636121
'''

# result = data[['width', 'height', 't40']]
#
# for row in result.values:
#     w = row[0] * 10  # 10베 축소하였기 다시 확
#     h = row[1] * 10
#     locate = row[2]
#     if w == 1000:
#         w = 999
#     if locate == 1:
#         cv2.circle(sprinkler,
#                    (int(h), int(w)),
#                    20, 1, -1)
#
# plt.imshow(sprinkler)
#
# # plt.imshow(land_house + land_tree + land_structure + land_facility)
# env_dict['sprinkler'] = [sprinkler, 5, [0, 0, 255]]
# env_map = env_dict['land'].copy()
# env_vis_map = np.zeros((env_map.shape[0], env_map.shape[1], 3))
# for component in ['house', 'structure', 'tree', 'facility', 'sprinkler']:
#     c_map = env_dict[component][0]
#     c_label = env_dict[component][1]
#     c_color = env_dict[component][2]
#     env_map[1 == c_map] = c_label
#     env_vis_map[1 == c_map, :] = np.array(c_color) / 255.
#
# env_dict['vis_map'] = env_vis_map
# plt.imshow(env_vis_map)
# plt.imsave('./data/result_fig_t40.jpg', env_vis_map)
