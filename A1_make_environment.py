import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
if not os.path.exists('./data'):
    os.mkdir('./data')

class tree:
    def __init__(self,
                 loc_dong=None,
                 id=None,
                 loc_w=None,
                 loc_h=None,
                 width=200,
                 shape='circle'):
        self.loc_dong = loc_dong
        self.id = id
        self.loc_w = loc_w
        self.loc_h = loc_h
        self.width = width
        self.shape = shape


class structure:
    def __init__(self,
                 loc_dong=None,
                 id=None,
                 loc_w=None,
                 loc_h=None,
                 width=5,
                 shape='circle'):
        self.loc_dong = loc_dong
        self.id = id
        self.loc_w = loc_w
        self.loc_h = loc_h
        self.width = width
        self.shape = shape


class facility:
    def __init__(self,
                 loc_dong=None,
                 id=None,
                 loc_w=None,
                 loc_h=None,
                 width=5,
                 height=5,
                 shape='circle'):
        self.loc_dong = loc_dong
        self.id = id
        self.loc_w = loc_w
        self.loc_h = loc_h
        self.width = width
        self.height = height
        self.shape = shape


# [width, height, offset from The most left protruding part,# offset from The most right protruding part] cm
dong = {'1': [500, 1500, 0, 0],
        '2': [500, 1500, 0, 0]}
rescale = 1

environment = {}
environment['dong'] = dong

dong_key = list(environment['dong'].keys())
dong_width = [];
dong_height = []
for i in dong_key:
    dong_width.append(environment['dong'][i][0])
    dong_height.append(environment['dong'][i][1] + \
                       environment['dong'][i][2] + \
                       environment['dong'][i][3])

#
land = np.zeros((sum(dong_width), max(dong_height)))
# plt.imshow(land)

land_house = land.copy()
width = 0
height = 0
for i in dong_key:
    dong_width, dong_height, offset_left, offset_right = environment['dong'][i]
    width += dong_width
    height = dong_height

    land_house[width - dong_width:width, offset_left:offset_left + height] = 1

# plt.imshow(land_house)
########
##tree##
########

tree_dict = {i: {} for i in dong_key}
##ctr + /
tree_info = {
    1: [[0, 140, 500, 200],
        [1, 170, 750, 200],
        [2, 160, 1100, 200],
        [3, 150, 1300, 200],
        [4, 350, 220, 200],
        [5, 340, 550, 200],
        [6, 370, 750, 200],
        [7, 360, 1080, 200],
        [8, 350, 1350, 200]],
    2: [[0, 150, 220, 200],
        [1, 140, 450, 200],
        [2, 170, 800, 200],
        [3, 160, 1080, 200],
        [4, 150, 1350, 200],
        [5, 350, 200, 200],
        [6, 340, 500, 200],
        [7, 370, 750, 200],
        [8, 360, 1100, 200],
        [9, 350, 1300, 200]
        ]
}
np.save('./data/tree_info.npy', tree_info)
for i in list(tree_info.keys()):
    for j in tree_info[i]:
        tree_dict[str(i)][j[0]] = tree(loc_dong=i, id=j[0], loc_w=j[1], loc_h=j[2], width=j[3])

land_tree = land.copy()

num_tree = len([t for i in dong_key for t in list(tree_dict[i].keys()) ])
land_tree_obj = np.zeros((num_tree, land_tree.shape[0], land_tree.shape[1]))
width = 0
height = 0

itr = 0
for i in dong_key:
    dong_width, dong_height, offset_left, offset_right = environment['dong'][i]
    width += dong_width
    height = dong_height

    for t in list(tree_dict[i].keys()):
        tree_loc_w = tree_dict[i][t].loc_w
        tree_loc_h = tree_dict[i][t].loc_h
        tree_width = tree_dict[i][t].width

        cv2.circle(land_tree,
                   (offset_left + tree_loc_h, width - dong_width + tree_loc_w),
                   tree_width // 2, 1, -1)
        cv2.circle(land_tree_obj[itr],
                   (offset_left + tree_loc_h, width - dong_width + tree_loc_w),
                   tree_width // 2, 1, -1)
        itr+=1
# plt.imshow(land_tree)
# plt.imshow(land_tree_obj[0])
########
##structure##
########


structure_range = {j: [i for i in range(0, dong[str(j)][1], 200)] + [dong[str(j)][1]] for j in list(tree_info.keys())}
structure_dict = {i: {} for i in dong_key}
##ctr + /
# id x, h, r
# structure_info = {i : [[0, 0, 0, 5] for j in structure_range[i]]}


structure_info = {
    1: [[0, 0, 0, 5],
        [0, 0, 200, 5],
        [0, 0, 400, 5],
        [0, 0, 600, 5],
        [0, 0, 800, 5],
        [0, 0, 1000, 5],
        [0, 0, 1200, 5],
        [0, 0, 1400, 5],
        [0, 0, 1500, 5],
        ],
    2: [[0, 500, 0, 5],
        [0, 500, 200, 5],
        [0, 500, 400, 5],
        [0, 500, 600, 5],
        [0, 500, 800, 5],
        [0, 500, 1000, 5],
        [0, 500, 1200, 5],
        [0, 500, 1400, 5],
        [0, 500, 1500, 5],
        ],
    3: [[0, 1000, 0, 5],
        [0, 1000, 200, 5],
        [0, 1000, 400, 5],
        [0, 1000, 600, 5],
        [0, 1000, 800, 5],
        [0, 1000, 1000, 5],
        [0, 1000, 1200, 5],
        [0, 1000, 1400, 5],
        [0, 1000, 1500, 5],
        ]
}
np.save('./data/structure_info.npy', structure_info)
structure_g_list = list(tree_info.keys()) + [max(list(tree_info.keys())) + 1]
structure_dict = {str(i): {} for i in structure_g_list}
for i in structure_g_list:
    for idx, j in enumerate(structure_info[i]):
        structure_dict[str(i)][idx] = structure(loc_dong=i, id=j[0], loc_w=j[1], loc_h=j[2], width=30)

land_structure = land.copy()
width = 0
height = 0
for i in dong_key:

    for t in list(structure_dict[i].keys()):
        tree_loc_w = structure_dict[i][t].loc_w
        tree_loc_h = structure_dict[i][t].loc_h
        tree_width = structure_dict[i][t].width

        cv2.circle(land_structure,
                   (offset_left + tree_loc_h, tree_loc_w),
                   tree_width // 2, 1, -1)

    if i == dong_key[-1]:
        ii = list(structure_dict.keys())[-1]
        for t in list(structure_dict[ii].keys()):
            tree_loc_w = structure_dict[ii][t].loc_w
            tree_loc_h = structure_dict[ii][t].loc_h
            tree_width = structure_dict[ii][t].width

            cv2.circle(land_structure,
                       (offset_left + tree_loc_h, tree_loc_w),
                       tree_width // 2, 1, -1)

# plt.imshow(land_tree + land_structure)


#### facility

facility_range = {j: [i for i in range(0, dong[str(j)][1], 200)] + \
                     [dong[str(j)][1]] for j in list(tree_info.keys())}
facility_dict = {i: {} for i in dong_key}
##ctr + /
# id x, h, r
# facility_info = {i : [[0, 0, 0, 5] for j in  facility_range[i]]}


facility_info = {
    1: [[0, 0, 0, 100, 50, 'rectangle'],
        ],
}

facility_dict = {str(i): {} for i in list(tree_info.keys())}
for i in list(facility_info.keys()):
    for idx, j in enumerate(facility_info[i]):
        facility_dict[str(i)][idx] = facility(loc_dong=i, id=j[0],
                                              loc_w=j[1], loc_h=j[2],
                                              width=j[3], height=j[4],
                                              shape=j[5])

land_facility = land.copy()
width = 0
height = 0
for i in dong_key:

    for t in list(facility_dict[i].keys()):
        _loc_w = facility_dict[i][t].loc_w
        _loc_h = facility_dict[i][t].loc_h
        _width = facility_dict[i][t].width
        _height = facility_dict[i][t].height
        _shape = facility_dict[i][t].shape

        if _shape == 'rectangle':
            cv2.rectangle(land_facility,
                          (offset_left + _loc_h, _loc_w),
                          (_height, _width),
                          1,
                          -1)
# plt.imshow(land_house + land_tree + land_structure + land_facility)
env_dict = {'land' : land,
            'house': [land_house, 1, [150, 75, 0]],
            'structure': [land_structure, 2, [255, 228, 0]],
            'tree': [land_tree, 3, [0, 255, 0]],
            'tree_obj':land_tree_obj,
            'facility': [land_facility, 4, [255, 0, 0]]
            }
env_map = env_dict['land'].copy()
env_vis_map = np.zeros((env_map.shape[0], env_map.shape[1], 3))
for component in ['house', 'structure', 'tree', 'facility']:
    c_map = env_dict[component][0]
    c_label = env_dict[component][1]
    c_color = env_dict[component][2]
    env_map[1 == c_map] = c_label
    env_vis_map[1 == c_map, :] = np.array(c_color) / 255.

env_dict['vis_map'] = env_vis_map
np.save('./data/env_dict.npy', env_dict)
plt.imshow(env_vis_map)

