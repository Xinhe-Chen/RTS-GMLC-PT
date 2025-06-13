import json
import os
import copy
import sys
import pyomo.environ as pyo
from fossil_npv_optimization import fossil_profit_opt
from utils import read_gmlc_gen, save_gen_data, make_lmp_csv

"""
Make the fossil generator parameters
"""
# df = read_gmlc_gen(csv_path="Data/gen.csv")
# save_gen_data(df, save_opt=True)

"""
Load the fossil generator parameters
"""
gen_dict_path = os.path.join(os.getcwd(), "Data", "gen_dict.json")
with open(gen_dict_path, "rb") as f:
    all_gen_dict = json.load(f)

"""
These part is for extracting LMPs from the bus_detail.csv
"""
# for idx, key in enumerate(list(fossil_gens.keys())):
#     if idx == 0:
#         make_lmp_csv(lmp_path=None, bus_details_path="Data/bus_detail.csv", bus_name=fossil_gens[key]["bus_name"])
#     else:
#         make_lmp_csv(lmp_path="Data/all_bus_lmp.csv", bus_details_path="Data/bus_detail.csv", bus_name=fossil_gens[key]["bus_name"])

"""
Select a generator for testing
"""
fossil_gens = copy.deepcopy(all_gen_dict["fossil"])
# bus_id = 101
gen_name = sys.argv[1]
gen_dict = fossil_gens[gen_name]
lmp_path = os.path.join("Data", "all_bus_lmp.csv")
m = fossil_profit_opt(gen_dict, lmp_path,)
# m.period[10].pprint()
# lmp = [pyo.value(getattr(m.period[t], "gen_"+gen_dict["name"]).LMP) for t in range(1,10)]
# print(f"LMP: {lmp}")

solver = pyo.SolverFactory("gurobi")
# solver.set_instance(m)
solver.options["MIPGap"] = 0.005
solver.solve(m, tee=True)

m.get_results("gen_" + gen_dict["name"]).to_csv(f"results/gen_{gen_dict["name"]}_result.csv", index=False)