import os
import json
import pyomo.environ as pyo
import pandas as pd
import idaes.logger as idaeslog
from idaes.apps.grid_integration import OperationModel, StochasticPriceTaker, RHPTForecaster
from util_gen_model_rolling_horizon import build_fossil_gen_design_model, build_fossil_gen_operation_model
from fossil_rolling_horizon_PT_parameter import gen_dict, period, scenario, horizon, planning_horizon, original_initial_state

_logger = idaeslog.getLogger(__name__)

def build_fossil_gen_flowsheet(m, params):
    """Builds the fossil generator flowsheet"""
    # build the design block

    # build the operational block
    for key in params:
        setattr(m, 
                params[key]["name"],
                OperationModel(
                    model_func=build_fossil_gen_operation_model,
                    model_args={"params": params[key]},
            )
        )
    m.power_to_grid = pyo.Var(within=pyo.NonNegativeReals)
    m.calculate_power_to_grid = pyo.Constraint(
        expr=m.power_to_grid == sum(getattr(m, params[key]["name"]).power for key in params.keys())
    )
    
    # here, assume all generators share the same LMP.
    m.elec_revenue = pyo.Expression(expr=getattr(m, params[key]["name"]).LMP * m.power_to_grid)


def fossil_profit_opt_scenario(forecaster, gen_dict):
    """Builds and returns an instance of the price-taker model"""

    m = StochasticPriceTaker(scenario, horizon, planning_horizon)

    # forecast the price signal at t = 0
    lmp_data = forecaster.forecast_prices(pointer=0)
    scenario_lmp_data = lmp_data[0]

    # just do one scenario for now
    scenario_model = m.build_PT_model(
        LMP_data=scenario_lmp_data,
        design_func=build_fossil_gen_design_model,
        design_params=gen_dict,
        flowsheet_func=build_fossil_gen_flowsheet,
        flowsheet_options={"params": gen_dict},
    )

    # Add operation limits
    scenario_model.add_capacity_limits(
        op_block_name="gen_" + gen_dict["name"],
        commodity="power",
        capacity=scenario_model.gen_design.gen_capacity,
        op_range_lb=gen_dict["min_p"]/gen_dict["max_p"],
    )

    # Add minimum uptime-downtime constraints on NGCC
    scenario_model.add_startup_shutdown(
        op_block_name="gen_" + gen_dict["name"],
        up_time=gen_dict["min_up_time"],
        down_time=gen_dict["min_down_time"],
    )

    # Add ramping constraints on NGCC
    scenario_model.add_ramping_limits(
        op_block_name="gen_" + gen_dict["name"],
        commodity="power",
        capacity=scenario_model.gen_design.gen_capacity,
        startup_rate=gen_dict["min_p"]/gen_dict["max_p"],
        shutdown_rate=gen_dict["min_p"]/gen_dict["max_p"],
        rampdown_rate=min(gen_dict["ramp"], gen_dict["max_p"])/gen_dict["max_p"],
        rampup_rate=min(gen_dict["ramp"], gen_dict["max_p"])/gen_dict["max_p"],
    )

    # Build, hourly cashflows, overall cashflows, and the objective function
    scenario_model.add_hourly_cashflows(
        revenue_streams=["elec_revenue"],
        operational_costs=None,
    )
    scenario_model.add_overall_cashflows(corporate_tax_rate=0)
    scenario_model.add_objective_function(objective_type="net_profit")

    return scenario_model


def fossil_profit_opt_stochastic(scenario, horizon, planning_horizon, forecaster, gen_dict, initial_state={}):
    """Builds and returns a stochastic price-taker model"""

    m = StochasticPriceTaker(scenario, horizon, planning_horizon, gen_dict=gen_dict)

    # forecast the price signal at t = 0
    lmp_data = forecaster.forecast_prices(pointer=0)

    design_func_dict = {}
    for key in list(gen_dict.keys()):
        design_func_dict[key] = build_fossil_gen_design_model

    # build scenario models
    scenario_model_list = m.generate_scenario_model_list(initial_state=initial_state,
                                                         LMP_data=lmp_data,
                                                         design_func_dict=design_func_dict,
                                                         design_params=gen_dict,
                                                         flowsheet_func=build_fossil_gen_flowsheet,
                                                         flowsheet_options={"params": gen_dict},
                                                         commodity="power",
                                                         operational_costs=["vom", "startup_cost", "shutdown_cost"],
                                                         )
    # check the lmp_data
    m.lmp_check(lmp_data)
    
    # build the stochastic price-taker model
    m.build_stochastic_PT_model(scenario_model_list=scenario_model_list, nonanti_varnames=["power_to_grid"])

    # Add objective function
    m.set_objective_function()

    return m


# read the LMP data
lmp_path = os.path.join("..", "Data", "all_bus_lmp.csv")
df_lmp = pd.read_csv(lmp_path)
lmp_data = df_lmp[gen_dict["gen_101_STEAM_3"]["bus_name"]+"_LMP"].to_numpy()

# define the scenario, horizon, and planning horizon

# define the forecaster
forecaster = RHPTForecaster(price_signal=lmp_data,
                            scenario=scenario,
                            horizon=horizon,
                            planning_horizon=planning_horizon)


"""
Build and solve single stochastic PT model and record results.
"""
# build the model
# stochastic_model = fossil_profit_opt_stochastic(scenario=scenario,
#                                                 horizon=horizon,
#                                                 planning_horizon=planning_horizon,
#                                                 forecaster=forecaster,
#                                                 gen_dict=gen_dict,
#                                                 initial_state=initial_state)

# check if the model is built successfully
# stochastic_model.pprint()
# print(stochastic_model._get_operation_vars(1, "power_to_grid"))

# # solve the model
# solver = "gurobi"
# opt_solver = pyo.SolverFactory(solver)
# soln = opt_solver.solve(stochastic_model, tee=True, options={"MIPGap": 0.01})

# _logger.info(f"Solver status: {soln.solver.status}")
# _logger.info(f"Termination condition: {soln.solver.termination_condition}")
# _logger.info(f"Objective value: {pyo.value(stochastic_model.obj)}")

# operation_var_name = ["op_mode", "power"]

# actual_price = forecaster.fetch_original_signal(pointer=0)

# res_dict = stochastic_model.record_solution(soln, actual_price=actual_price, operation_var_name=operation_var_name)

# # save the results
# with open(f"results/test_gen_{gen_dict['name']}_result.json", "w") as f:
#     json.dump(res_dict, f)
# print(res_dict)


"""
Build and solve a rolling horizon stochastic PT model and record results.
"""
solver = "gurobi"
opt_solver = pyo.SolverFactory(solver)
results_dict = {}
operation_var_name = ["op_mode", "power"]
initial_state = original_initial_state
for i in range(0, period):
    _logger.info(f"Building price-taker optimization for period {i}.")
    stochastic_model = fossil_profit_opt_stochastic(scenario=scenario,
                                                horizon=horizon,
                                                planning_horizon=planning_horizon,
                                                forecaster=forecaster,
                                                gen_dict=gen_dict,
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

with open(f"results/test_{period}_gen_101_STEAM_3_result.json", "w") as f:
    json.dump(results_dict, f)
print(results_dict)
