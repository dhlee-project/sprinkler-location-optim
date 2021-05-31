
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
data = np.load('./data/env_dict.npy', allow_pickle=True).item()
tree_info = np.load('./data/tree_info.npy', allow_pickle=True).item()

##  tree info
dong_list = list(tree_info.keys())
tree_list = []
num = 0
for i in dong_list:
    for j in tree_info[i]:
        tree_list.append([num, j[1]//10, j[2]//10])
        num += 1

##  map resize
tree_map = data['tree'][0]
t_width = tree_map.shape[0]
t_height = tree_map.shape[1]
tree_map2 = cv2.resize(tree_map, dsize=(int(t_height*0.1),int(t_width*0.1)))

## tree onject map resize
num_tree = data['tree_obj'].shape[0]
tree_obj_arr = np.zeros((num_tree, int(t_width*0.1), int(t_height*0.1)))
for i in range(num_tree):
    to_width = tree_map.shape[0]
    to_height = tree_map.shape[1]
    tree_obj_arr[i] = cv2.resize(data['tree_obj'][i], dsize=(int(to_height*0.1), int(to_width*0.1)))

# assign weight value. tree : 5 otherwise : 1
tree_map2 = (tree_map2 * 4) + 1 ### 가중치

# calculate sprinkler value
s_width = 500 // 10
s_height = 200 // 10
s_range = 300 // 10

# tree의 수분 공급량
tree_water = np.zeros((num_tree, 2))

# x 만들기
sprinkler_value_dict = {f'X_{i}_{j}': [i, j, 0] + [0]*num_tree + [0]*num_tree for i in [0, 50, 100] for j in range(150)}
tree_dict_key = [f'A_{i[1]}_{i[2]}' for i in tree_list] + [f'C_{i[1]}_{i[2]}' for i in tree_list]

#tree_value_dict = {f'Y_{i[1]}_{i[2]}_X_{j}_{k}' : [0,0] for i in tree_list for j in [0, 50, 100] for k in range(150)}


import tqdm
datatable = np.zeros((len(list(sprinkler_value_dict.keys())), 3+len(tree_dict_key)))
datatable.shape
itr = 0
for i in list(sprinkler_value_dict.keys()):
    s_width, s_height = sprinkler_value_dict[i][:2]

    # 스프링쿨러
    sprinkler_tmp = np.zeros_like(tree_map2)

    # 스프링클러 범위 그리기
    cv2.circle(sprinkler_tmp,
               (s_height, s_width),
               s_range, 1, -1)
    value = tree_map2 * sprinkler_tmp
    sprinkler_value_dict[i][2] = np.sum(value)

    for j in range(num_tree):
        tree_obj = tree_obj_arr[j]
        total_area = np.sum(tree_obj != 0)
        inter_area = np.sum(tree_obj == (sprinkler_tmp * 2) - 1)

        sprinkler_value_dict[i][3 + j] += inter_area
        if inter_area/total_area > 0.8:
            sprinkler_value_dict[i][3 + 19 + j] += 1
        datatable[itr][:] = sprinkler_value_dict[i]
    itr += 1

index_name = list(sprinkler_value_dict.keys())
col_name = ['width', 'height', 'value'] + tree_dict_key
datatable = pd.DataFrame(datatable, columns=col_name, index=index_name)
datatable.to_csv('./data/value_table.csv')