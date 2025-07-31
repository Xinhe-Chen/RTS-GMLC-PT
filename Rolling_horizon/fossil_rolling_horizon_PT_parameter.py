import os
import copy
import json
import pandas as pd

"""
Read the generator parameters
"""
gen_dict_path = os.path.join(os.getcwd(), "..", "Data", "gen_dict.json")
with open(gen_dict_path, "rb") as f:
    all_gen_dict = json.load(f)
fossil_gens = copy.deepcopy(all_gen_dict["fossil"])

"""
Reform the gen_dict to be used in the model.
"""
gen_dict = {}
# We can either define a single generator or multiple generators/IESs.
gen_names = ["101_STEAM_3"]
# gen_names = ["101_STEAM_3", "101_CT_1"]
for gen_name in gen_names:
    individual_gen_dict = fossil_gens[gen_name]
    individual_gen_dict["name"] = "gen_" + individual_gen_dict["name"]
    gen_dict[individual_gen_dict["name"]] = individual_gen_dict

# read the LMP data
lmp_path = os.path.join("..", "Data", "all_bus_lmp.csv")
df_lmp = pd.read_csv(lmp_path)
lmp_data = df_lmp[gen_dict["gen_101_STEAM_3"]["bus_name"]+"_LMP"].to_numpy()

# make a pseduo initial state
initial_state_1 = {
    "name": list(gen_dict.keys())[0],
    "up_time": 0,
    "down_time": 10,
    "min_up_time": gen_dict[list(gen_dict.keys())[0]]["min_up_time"],
    "min_down_time": gen_dict[list(gen_dict.keys())[0]]["min_down_time"],
}
# initial_state_2 = {
#     "name": list(gen_dict.keys())[1],
#     "up_time": 0,
#     "down_time": 10,
#     "min_up_time": gen_dict[list(gen_dict.keys())[0]]["min_up_time"],
#     "min_down_time": gen_dict[list(gen_dict.keys())[0]]["min_down_time"],
# }

initial_state_list = [initial_state_1]
original_initial_state = {}
for idx, key in zip(range(len(gen_dict)), gen_dict.keys()):
    original_initial_state[key] = initial_state_list[idx]

period = 14
scenario, horizon, planning_horizon = 5, 36, 24