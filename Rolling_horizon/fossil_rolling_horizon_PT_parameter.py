import os
import copy
import json

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
gen_names = [i for i in fossil_gens.keys()]  # all fossil generators
for gen_name in gen_names:
    individual_gen_dict = fossil_gens[gen_name]
    individual_gen_dict["name"] = "gen_" + individual_gen_dict["name"]
    gen_dict[individual_gen_dict["name"]] = individual_gen_dict

# make a pseduo initial state
original_initial_state = {}
for idx, key in zip(range(len(gen_dict)), gen_dict.keys()):
    initial_state = {
        "name": key,
        "up_time": 0,
        "down_time": 100,
        "min_up_time": gen_dict[key]["min_up_time"],
        "min_down_time": gen_dict[key]["min_down_time"],
    }
    # initial_state_2 = {
    #     "name": list(gen_dict.keys())[1],
    #     "up_time": 0,
    #     "down_time": 10,
    #     "min_up_time": gen_dict[list(gen_dict.keys())[0]]["min_up_time"],
    #     "min_down_time": gen_dict[list(gen_dict.keys())[0]]["min_down_time"],
    # }
    original_initial_state[key] = initial_state

period = 366
scenario, horizon, planning_horizon = 5, 36, 24