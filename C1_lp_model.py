import numpy as np
import pandas as pd
import pulp
import cv2
import matplotlib.pyplot as plt
############################################
# Load DATA
data = pd.read_csv('./data/value_table.csv', index_col=0)
structure_info = np.load('./data/structure_info.npy', allow_pickle=True).item()


############################################
# Define MODEL
print("== for sprinkler_model")
sprinkler_model = pulp.LpProblem(
    name="sprinkler_model",
    sense=pulp.LpMaximize
)

############################################
# Define VARIABLE

X_index = data.index.values

vars_x = pulp.LpVariable.dicts(
    name='sprk',  # prefix of each LP var
    indexs=X_index,
    lowBound=0,
    upBound=1,
    cat='Binary')

Y_index = data.columns.values[3:3+19]

vars_y = pulp.LpVariable.dicts(
    name='Y',  # prefix of each LP var
    indexs=Y_index,
    lowBound=0,
    cat='Integer'
)

Z_index = data.columns.values[3+19:3+19+19]

vars_z = pulp.LpVariable.dicts(
    name='Z',  # prefix of each LP var
    indexs=Z_index,
    lowBound=0,
    cat='Integer'
)

############################################
# Define OBJECTIVE function
coefficient = data['value'].values
obj_function = pulp.lpSum(c * v for c, v in zip(coefficient, vars_x.values()))
sprinkler_model.objective = obj_function
#print(sprinkler_model.objective)

############################################
# Define CONSTRAINT function
# 최소 설치간격
# 100cm

constraint_1 = []

bound = 10
w_list = np.array([int(i.split('_')[1]) for i in X_index]).astype(int)
h_list = np.array([int(i.split('_')[2]) for i in X_index]).astype(int)

x_tmp = X_index[0]
for x_tmp in X_index:
    var_prefix, w, h = x_tmp.split('_')
    lower = int(h) - bound
    upper = int(h) + bound
    w_mask = w_list == int(w)
    h_lo_mask = h_list >= lower
    h_up_mask = h_list <= upper
    l1 = np.logical_and(h_lo_mask, h_up_mask)
    l2 = np.logical_and(l1, w_mask)

    g = None
    for i in X_index[l2]:
        g += vars_x[i]
    constraint_1.append(g <= 1)


# 모든 나무에는 하나 이상의 스프링클러가 설치되어있어야함
# 80%
constraint_2 = []
for z_tmp in Z_index:
    coefficient = data[z_tmp].values
    g = pulp.lpSum(c * v for c, v in zip(coefficient, vars_x.values()))
    constraint_2.append(g >= 1)

# constraint_3 = []
# g = pulp.lpSum(v for v in zip(vars_x.values()))
# constraint_3.append(g <= 8)

constraint_4 = []
structure_loc = [f'X_{j[1]//10}_{j[2]//10}' for i in list(structure_info.keys()) for j in structure_info[i]]

bound = 3
w_list = np.array([int(i.split('_')[1]) for i in X_index]).astype(int)
h_list = np.array([int(i.split('_')[2]) for i in X_index]).astype(int)

for x_tmp in structure_loc:
    var_prefix, w, h = x_tmp.split('_')
    lower = int(h) - bound
    upper = int(h) + bound
    w_mask = w_list == int(w)
    h_lo_mask = h_list >= lower
    h_up_mask = h_list <= upper
    l1 = np.logical_and(h_lo_mask, h_up_mask)
    l2 = np.logical_and(l1, w_mask)

    g = None
    for i in X_index[l2]:
        g += vars_x[i]
    constraint_4.append(g == 0)

# 물의 공급량 제약
constraint_5 = []
constraint_6 = []
for y_tmp in Y_index:
    coefficient = data[y_tmp].values
    g = pulp.lpSum(c * v for c, v in zip(coefficient, vars_x.values()))
    constraint_5.append(g>=300)
    constraint_6.append(g<=340)
#

############################################
# Add Constraint
## add constraint into model
constraints = constraint_1 + constraint_2 + constraint_4 + constraint_5  + constraint_6
for i, c in enumerate(constraints):
    constraint_name = f"c_{i}"
    sprinkler_model.constraints[constraint_name] = c

############################################
# Solve MODEL
## solve problem
sprinkler_model.solve()
# 잘 풀렸는지 확인, infeasible 등이 없는지 확인할 것.
print("Status:", pulp.LpStatus[sprinkler_model.status])
#for v in sprinkler_model.variables():
#    print(f"Produce {v.varValue:5.1f} Cake {v}")

## arrange result as dictionary
#for v in LPmodel_complex.variables():
#    print(f"{v}:{v.value():5.1f}")
result_dict = {str(v)[5:] : int(v.value()) for v in sprinkler_model.variables()}


result = data[['width', 'height', 'value']]
result.loc[: , 'locate'] = 0
for k in result_dict:
    result.loc[k, 'locate'] = result_dict[k]
result.to_csv('./data/result_300_340.csv')


############################################
# Visualization

env_dict = np.load('./data/env_dict.npy', allow_pickle=True).item()
sprinkler = env_dict['land'].copy()

for row in result.values:
    w = row[0] * 10 # 10베 축소하였기 다시 확
    h = row[1] * 10
    locate = row[3]
    if w == 1000:
        w = 999
#    sprinkler[int(w), int(h)] = locate
    if locate == 1:
        cv2.circle(sprinkler,
                   (int(h), int(w)),
                   20, 1, -1)

# plt.imshow(land_house + land_tree + land_structure + land_facility)
env_dict['sprinkler'] = [sprinkler, 5, [0, 0, 255]]

env_map = env_dict['land'].copy()
env_vis_map = np.zeros((env_map.shape[0], env_map.shape[1], 3))
for component in ['house', 'structure', 'tree', 'facility', 'sprinkler']:
    c_map = env_dict[component][0]
    c_label = env_dict[component][1]
    c_color = env_dict[component][2]
    env_map[1 == c_map] = c_label
    env_vis_map[1 == c_map, :] = np.array(c_color) / 255.

env_dict['vis_map'] = env_vis_map
# np.save('./data/env_dict_reuslt.npy', env_dict)
plt.imshow(env_vis_map)
plt.imsave('./data/result_fig_300_340.jpg',env_vis_map)


