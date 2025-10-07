import os
import sys
import json
import pyomo.environ as pyo
import pandas as pd
import idaes.logger as idaeslog
from idaes.apps.grid_integration import RHPTForecaster
from fossil_rolling_horizon_PT_parameter import gen_dict, period, scenario, horizon, planning_horizon, original_initial_state
from fossil_rolling_horizon_optimization import fossil_profit_opt_stochastic

_logger = idaeslog.getLogger(__name__)
if len(sys.argv) > 1:
    gen_name = sys.argv[1]
    print(f"The generator is: {gen_name}")
else:
    print("No generator provided. Set to default: 101_STEAM_3.")
    gen_name = "101_STEAM_3"

specific_gen_dict = {gen_name: gen_dict["gen_" + gen_name]}
# read the LMP data
# lmp_path = os.path.join("..", "Data", "all_bus_lmp.csv")
lmp_path = os.path.join("..", "Notebook", "Bus_LMP.csv")
df_lmp = pd.read_csv(lmp_path)
lmp_data = df_lmp[specific_gen_dict[gen_name]["bus_name"]+"_LMP"].to_numpy()

# define the forecaster
forecaster = RHPTForecaster(price_signal=lmp_data,
                            scenario=scenario,
                            horizon=horizon,
                            planning_horizon=planning_horizon)


"""
Build and solve a rolling horizon stochastic PT model and record results.
"""
solver = "gurobi"
opt_solver = pyo.SolverFactory(solver)
results_dict = {}
operation_var_name = ["op_mode", "power"]
initial_state = {gen_name: original_initial_state["gen_" + gen_name]}
for i in range(0, period):
    _logger.info(f"Building price-taker optimization for period {i}.")
    # forecast the price signal at t = 0
    lmp_data = forecaster.forecast_prices(pointer=i)
    # build the stochastic model
    stochastic_model = fossil_profit_opt_stochastic(scenario=scenario,
                                                horizon=horizon,
                                                planning_horizon=planning_horizon,
                                                lmp_data=lmp_data,
                                                gen_dict=specific_gen_dict,
                                                initial_state=initial_state)
    soln = opt_solver.solve(stochastic_model, tee=True, options={"MIPGap": 0.01})
    
    _logger.info(f"Solver status: {soln.solver.status}")
    _logger.info(f"Termination condition: {soln.solver.termination_condition}")
    _logger.info(f"Objective value: {pyo.value(stochastic_model.obj)}")

    actual_price = forecaster.fetch_original_signal(pointer=i)
    res_dict = stochastic_model.record_solution(soln, actual_price=actual_price, operation_var_name=operation_var_name)

    # update initial_state
    initial_state = stochastic_model.report_final_state()
    results_dict[f"period_{i}"] = res_dict

# save results
with open(f"results/test_{period}_{gen_name}_result.json", "w") as f:
    json.dump(results_dict, f)
# print(results_dict)