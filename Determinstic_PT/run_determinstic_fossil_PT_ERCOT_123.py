import os
import json
import pandas as pd
import pyomo.environ as pyo
import idaes.logger as idaeslog
from determinstic_fossil_PT_opt import determinstic_fossil_profit_opt
from pyomo.opt import TerminationCondition

_logger = idaeslog.getLogger(__name__)

"""
This script runs the sweep of deterministic profit optimization for a fossil fuel generator
in the ERCOT 123 bus system.
"""

# read the generartor parameters from the json file
gen_path = os.path.join(os.getcwd(), "..", "Data", "gen_dict_ercot_123.json")
with open(gen_path, "r") as f:
    gen_dict = json.load(f)

# read the LMP data from the csv file
lmp_path = os.path.join(os.getcwd(), "..", "Notebook", "Bus_LMP.csv")
df_lmp = pd.read_csv(lmp_path)

# read the dispatch data from the csv file if needed
dispatch_path = os.path.join(os.getcwd(), "..", "Notebook", "Generator_Dispatch.csv")
df_dispatch = pd.read_csv(dispatch_path)


def save_results(m, gen_name):
    """
    Check the results of the optimization.
    """
    result_dict = {}
    result_dict["objective"] = pyo.value(m.obj)
    for p in m.period:
        result_dict[p[1]] = {}
        result_dict[p]["power"] = pyo.value(m.period[p].power_to_grid)
        result_dict[p[1]]["rev"] = pyo.value(m.period[p].elec_revenue)
        result_dict[p[1]]["vom"] = pyo.value(getattr(m.period[p], gen_name).vom)
        result_dict[p[1]]["startup"] = pyo.value(getattr(m.period[p], gen_name).startup)

    return result_dict


def build_PT_model(gen_name, solve=True):
    """
    Build the determinstic price-taker model for the profit optimization of a fossil fuel generator.

    Args:
        gen_name (str): The name of the generator to be optimized.
    """
    # Get the parameters for the specified generator
    params = gen_dict["fossil"][gen_name]

    # Get the LMP data for the generator
    lmp_data_all = df_lmp[params["bus_name"] + "_LMP"].to_numpy()
    lmp_data = lmp_data_all.copy()

    # Get the dispatch data for the generator
    dispatch_data = df_dispatch[params["gen_name"] + "_Dispatch"].to_numpy()

    # Build the optimization model
    m = determinstic_fossil_profit_opt(params, lmp_data, dispatch_data, fixing_dispatch=False)

    # solve the model
    if solve:
        solver = pyo.SolverFactory("gurobi_persistent")
        solver.set_instance(m)
        solver.options["MIPGap"] = 0.005
        result = solver.solve(tee=True)

        # save the results
        res_dict = save_results(m)
        with open(f"ERCOT_123_fossil_{gen_name}_det_PT_results.json", "w") as f:
            json.dump(res_dict, f)

    return m, result

# # if you want to check the results of a specific generator, uncomment the following line
# m.period[1,16].pprint()
# m.gen_101_STEAM_3_startup_shutdown.pprint()

# build the for loop for sweeping through all the fossil generators in the gen_dict
for gen_name in gen_dict["fossil"]:
    _logger.info(f"Building and solving model for generator: {gen_name}")
    m, result = build_PT_model(gen_name, solve=True)
    # checkt the solver status
    if result.solver.termination_condition == TerminationCondition.optimal:
        _logger.info(f"Optimal solution found for generator: {gen_name}")
    elif result.solver.termination_condition == TerminationCondition.infeasible:
        _logger.warning(f"Infeasible solution for generator: {gen_name}")
    else:
        _logger.warning(f"Solver ended with condition {result.solver.termination_condition} for generator: {gen_name}")