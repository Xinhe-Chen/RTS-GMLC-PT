import sys
import os
import json
import copy
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pyomo.environ as pyo
import pandas as pd
import idaes.logger as idaeslog
from idaes.apps.grid_integration import DesignModel, OperationModel
from idaes.apps.grid_integration import StochasticPriceTaker
from idaes.apps.grid_integration import RHPTForecaster
from util_gen_model_rolling_horizon import build_gen_design_model, build_fossil_gen_operation_model

_logger = idaeslog.getLogger(__name__)


def build_fossil_gen_flowsheet(m, gen_dict):
    """Builds the fossil generator flowsheet"""

    setattr(m, 
            gen_dict["name"],
            OperationModel(
                model_func=build_fossil_gen_operation_model,
                model_args={"gen_dict": gen_dict},
        )
    )
    m.power_to_grid = pyo.Var(within=pyo.NonNegativeReals)
    m.calculate_power_to_grid = pyo.Constraint(
        expr=m.power_to_grid == getattr(m, gen_dict["name"]).power
    )
    m.elec_revenue = pyo.Expression(expr=getattr(m, gen_dict["name"]).LMP * m.power_to_grid)


def fossil_profit_opt_scenario(forecaster, gen_dict):
    """Builds and returns an instance of the price-taker model"""

    m = StochasticPriceTaker(scenario, horizon, planning_horizon)

    # forecast the price signal at t = 0
    lmp_data = forecaster.forecast_prices(pointer=0)
    scenario_lmp_data = lmp_data[0]

    # just do one scenario for now
    scenario_model = m.build_PT_model(
        LMP_data=scenario_lmp_data,
        design_func=build_gen_design_model,
        gen_dict=gen_dict,
        flowsheet_func=build_fossil_gen_flowsheet,
        flowsheet_options={"gen_dict": gen_dict},
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

    # build the stochastic price-taker model
    m.build_stochastic_PT_model(
        initial_state=initial_state,
        LMP_data=lmp_data,
        design_func=build_gen_design_model,
        flowsheet_func=build_fossil_gen_flowsheet,
        flowsheet_options={"gen_dict": gen_dict},
        nonanti_varnames=["power_to_grid"],
        operational_costs=["vom", "startup_cost", "shutdown_cost"],
    )

    # Add objective function
    m.set_objective_function()

    return m

# read the generator parameters
gen_dict_path = os.path.join(os.getcwd(), "..", "Data", "gen_dict.json")
with open(gen_dict_path, "rb") as f:
    all_gen_dict = json.load(f)
fossil_gens = copy.deepcopy(all_gen_dict["fossil"])
gen_name = "101_STEAM_3"
gen_dict = fossil_gens[gen_name]
gen_dict["name"] = "gen_" + gen_dict["name"]

# read the LMP data
lmp_path = os.path.join("..", "Data", "all_bus_lmp.csv")
df_lmp = pd.read_csv(lmp_path)
lmp_data = df_lmp[gen_dict["bus_name"]+"_LMP"].to_numpy()

# define the scenario, horizon, and planning horizon
scenario, horizon, planning_horizon = 5, 36, 24
# define the forecaster
forecaster = RHPTForecaster(price_signal=lmp_data,
                            scenario=scenario,
                            horizon=horizon,
                            planning_horizon=planning_horizon)

# build the fossil generator profit optimization model
# scenario_model = fossil_profit_opt(forecaster, gen_dict)
# scenario_model.pprint()

# build stochastic price-taker model
initial_state = {
    "name": gen_dict["name"],
    "up_time": 0,
    "down_time": 10,
    "min_up_time": gen_dict["min_up_time"],
    "min_down_time": gen_dict["min_down_time"],
}

stochastic_model = fossil_profit_opt_stochastic(scenario=scenario,
                                                horizon=horizon,
                                                planning_horizon=planning_horizon,
                                                forecaster=forecaster,
                                                gen_dict=gen_dict,
                                                initial_state=initial_state)
# stochastic_model.pprint()
# print(stochastic_model._get_operation_vars(1, "power_to_grid"))
solver = "gurobi"
opt_solver = pyo.SolverFactory(solver)
soln = opt_solver.solve(stochastic_model, tee=True, options={"MIPGap": 0.01})

_logger.info(f"Solver status: {soln.solver.status}")
_logger.info(f"Termination condition: {soln.solver.termination_condition}")
_logger.info(f"Objective value: {pyo.value(stochastic_model.obj)}")

operation_var_name = ["op_mode", "power"]

actual_price = forecaster.fetch_original_signal(pointer=0)

res_dict = stochastic_model.record_solution(soln, actual_price=actual_price, operation_var_name=operation_var_name)
with open(f"results/test_gen_{gen_dict['name']}_result.json", "w") as f:
    json.dump(res_dict, f)
print(res_dict)