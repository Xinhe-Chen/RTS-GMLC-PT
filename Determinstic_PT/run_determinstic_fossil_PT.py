import os
import json
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import idaes.logger as idaeslog
from determinstic_fossil_PT_opt import determinstic_fossil_profit_opt


_logger = idaeslog.getLogger(__name__)

# read the generartor parameters from the json file
gen_path = os.path.join(os.getcwd(), "..", "Data", "gen_dict.json")

with open(gen_path, "r") as f:
    gen_dict = json.load(f)

gen_name = "101_STEAM_3"

params = gen_dict["fossil"][gen_name]


# read the LMP data from the csv file
lmp_path = os.path.join(os.getcwd(), "..", "Data", "all_bus_lmp.csv")

df_lmp = pd.read_csv(lmp_path)

lmp_data_all = df_lmp[params["bus_name"] + "_LMP"].to_numpy()
lmp_data = lmp_data_all[0:24]

# run the optimization
m = determinstic_fossil_profit_opt(params, lmp_data)

m.pprint()

# solver = pyo.SolverFactory("gurobi_persistent")
# solver.set_instance(m)
# solver.options["MIPGap"] = 0.005
# result = solver.solve(tee=True)