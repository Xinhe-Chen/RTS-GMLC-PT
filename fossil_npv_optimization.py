import numpy as np
import pandas as pd
import pyomo.environ as pyo
import idaes.logger as idaeslog
from idaes.apps.grid_integration import DesignModel, OperationModel, PriceTakerModel
from idaes.core.util.config import ConfigurationError
from util_gen_model import build_gen_design_model, build_fossil_gen_operation_model
from utils import gen_startup_cost, ramping_limits, startup_shutdown_constraints
from general_gen_model import GmlcGen, PriceTakerRTSGMLC

_logger = idaeslog.getLogger(__name__)


def build_fossil_gen_flowsheet(m, gen_des_blk, gen_dict):
    """Builds the fossil generator flowsheet"""

    setattr(m, 
            "gen_" + gen_dict["name"],
            OperationModel(
                model_func=build_fossil_gen_operation_model,
                model_args={"design_blk": gen_des_blk, "gen_dict": gen_dict},
        )
    )

    m.power_to_grid = pyo.Var(within=pyo.NonNegativeReals)
    m.calculate_power_to_grid = pyo.Constraint(
        expr=m.power_to_grid == getattr(m, "gen_" + gen_dict["name"]).power
    )
    m.elec_revenue = pyo.Expression(expr=getattr(m, "gen_"+gen_dict["name"]).LMP * m.power_to_grid)


def fossil_profit_opt(gen_dict, lmp_path, configuration=None):
    """Builds and returns an instance of the price-taker model"""
    m = PriceTakerRTSGMLC(gen_dict=gen_dict, lmp_path=lmp_path)

    # Appending the data to the model
    # m.append_lmp_data(lmp_data=lmp_data)

    # Build design models and fix the capacity
    m.gen_design = DesignModel(
        model_func=build_gen_design_model,
        model_args={"gen_dict": gen_dict},
    )

    # Build multiperiod operation model
    m.build_multiperiod_model(
        flowsheet_func=build_fossil_gen_flowsheet,
        flowsheet_options={
            "gen_des_blk": m.gen_design,
            "gen_dict": gen_dict,
        },
    )

    # Define useful expressions
    # m.total_co2_produced = pyo.Expression(
    #     expr=sum(m.period[p].ngcc.co2_emissions for p in m.period)
    # )

    # Add operation limits
    m.add_capacity_limits(
        op_block_name="gen_" + gen_dict["name"],
    )

    # Add minimum uptime-downtime constraints on NGCC
    m.add_startup_shutdown(
        op_block_name="gen_" + gen_dict["name"],
    )

    # Add ramping constraints on NGCC
    m.add_ramping_limits(
        op_block_name="gen_" + gen_dict["name"],
    )

    # Build, hourly cashflows, overall cashflows, and the objective function
    m.add_hourly_cashflows(
        op_block_name="gen_" + gen_dict["name"],
        revenue_streams=["elec_revenue"],
        operational_costs=["vom"],
    )

    m.add_overall_cashflows(corporate_tax_rate=0)
    m.add_objective_function(objective_type="net_profit")

    return m