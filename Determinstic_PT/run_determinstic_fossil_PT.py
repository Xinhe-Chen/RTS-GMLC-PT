import os
import json
import pandas as pd
import pyomo.environ as pyo
import idaes.logger as idaeslog
from determinstic_fossil_PT_opt import determinstic_fossil_profit_opt
from pyomo.opt import TerminationCondition


_logger = idaeslog.getLogger(__name__)

# read the generartor parameters from the json file
gen_path = os.path.join(os.getcwd(), "..", "Data", "gen_dict.json")

with open(gen_path, "r") as f:
    gen_dict = json.load(f)

gen_name = "101_STEAM_3"

params = gen_dict["fossil"][gen_name]


# read the LMP data from the csv file
lmp_path = os.path.join(os.getcwd(), "..", "Notebook", "Bus_LMP.csv")

df_lmp = pd.read_csv(lmp_path)

lmp_data_all = df_lmp[params["bus_name"] + "_LMP"].to_numpy()
# lmp_data = lmp_data_all[0:24]
lmp_data = lmp_data_all.copy()

dispatch_path = os.path.join(os.getcwd(), "..", "Notebook", "Generator_Dispatch.csv")
df_dispatch = pd.read_csv(dispatch_path)
dispatch_data = df_dispatch["101_STEAM_3_Dispatch"].to_numpy()

# run the optimization
_logger.info(f"Building and solving model for generator: {gen_name}")
m = determinstic_fossil_profit_opt(params, lmp_data, dispatch_data, fixing_dispatch=False)

# m.period[1,16].pprint()
# m.gen_101_STEAM_3_startup_shutdown.pprint()

solver = pyo.SolverFactory("gurobi_persistent")
solver.set_instance(m)
solver.options["MIPGap"] = 0.005
result = solver.solve(tee=True)

# checkt the solver status
if result.solver.termination_condition == TerminationCondition.optimal:
    _logger.info(f"Optimal solution found for generator: {gen_name}")
elif result.solver.termination_condition == TerminationCondition.infeasible:
    _logger.warning(f"Infeasible solution for generator: {gen_name}")
else:
    _logger.warning(f"Solver ended with condition {result.solver.termination_condition} for generator: {gen_name}")

def save_results(m):
    """
    Check the results of the optimization.
    """
    result_dict = {}
    result_dict["objective"] = pyo.value(m.obj)
    for p in m.period:
        result_dict[p[1]] = {}
        # result_dict[p]["power"] = pyo.value(m.period[p].power_to_grid)
        result_dict[p[1]]["rev"] = pyo.value(m.period[p].elec_revenue)
        result_dict[p[1]]["vom"] = pyo.value(m.period[p].gen_101_STEAM_3.vom)
        result_dict[p[1]]["startup"] = pyo.value(m.period[p].gen_101_STEAM_3.startup)
    return result_dict

with open(f"det_fossil_{gen_name}_PT_unfixed_dispatch_results.json", "w") as f:
    json.dump(save_results(m), f)