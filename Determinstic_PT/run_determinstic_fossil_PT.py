import os
import json
import pandas as pd
import pyomo.environ as pyo
import idaes.logger as idaeslog
from determinstic_fossil_PT_opt import determinstic_fossil_profit_opt
from pyomo.util.infeasible import (
    log_infeasible_constraints,
    log_infeasible_bounds,
    log_close_to_bounds,
)


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
# lmp_data = lmp_data_all[0:24]
lmp_data = lmp_data_all.copy()

dispatch_path = os.path.join(os.getcwd(), "..", "Notebook", "Generator_Dispatch.csv")
df_dispatch = pd.read_csv(dispatch_path)
dispatch_data = df_dispatch["101_STEAM_3_Dispatch"].to_numpy()

# run the optimization
m = determinstic_fossil_profit_opt(params, lmp_data, dispatch_data, fixing_dispatch=True)

# m.period[1,16].pprint()
# m.gen_101_STEAM_3_startup_shutdown.pprint()

solver = pyo.SolverFactory("gurobi_persistent")
solver.set_instance(m)
solver.options["MIPGap"] = 0.005
result = solver.solve(tee=True)

log_infeasible_bounds(result, tol=1e-8)
log_infeasible_constraints(result, tol=1e-8)
log_close_to_bounds(result, tol=1e-8)   # useful to see tight/active bounds

# def save_results(result):
#     """
#     Check the results of the optimization.
#     """
#     result_dict = {}
#     result_dict["objective"] = pyo.value(result.objective)
#     for p in result.period:
#         result_dict[p] = {}
#         result_dict[p]["power"] = pyo.value(result.period[p].power)
#         result_dict[p]["vom"] = pyo.value(result.period[p].vom)
#         result_dict[p]["startup_cost"] = pyo.value(result.period[p].startup_cost)
#     return result_dict

# with open("det_fossil_PT_fixed_dispatch_results.json", "w") as f:
#     json.dump(save_results(result), f)