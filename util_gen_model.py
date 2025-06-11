import numpy as np
import pandas as pd
import pyomo.environ as pyo
import idaes.logger as idaeslog

_logger = idaeslog.getLogger(__name__)

def build_gen_design_model(m, gen_dict):
    '''
    build generator design models. 
    We do not do design optimization in this work so the maximun capacity is a parameter.  
    
    Args:
        m: pyomo model.
        gen_dict: dictionary that stores the generator parameters.

    Returns:
        None
    '''
    # if fom and capex not provide, just set them to 0
    if "capex" not in list(gen_dict.keys()):
        gen_dict["capex"] = 0

    if "fom" not in list(gen_dict.keys()):
        gen_dict["fom"] = 0

    m.gen_capacity = pyo.Param(
        initialize=gen_dict['max_p'],
        mutable=True,
        doc="Maxium capacity of the generator [in MW]",
    )
    m.capex = pyo.Expression(
        expr=m.gen_capacity * gen_dict["capex"],
        )
    
    m.fom = pyo.Expression(
        expr=m.gen_capacity * gen_dict["fom"]
    )
    
    return


def build_fossil_gen_operation_model(m, design_blk, gen_dict):
    """
    Function that adds the fossil generator operation model

    Args:
        m: Pyomo Block
        design_blk: Pyomo block containing the design model
        gen_dict: dictionary that stores the generator parameters.

    Returns:
        None
    """
    # the power output at each time period
    m.power = pyo.Var(
        within=pyo.NonNegativeReals,
        doc="Net power produced by NGCC at time t [in MW]",
        bounds=(0, design_blk.gen_capacity.value),
        # doc="Output of the power at time t"
    )
    # placeholder: theorically, we can calculate the CO2 emission.

    # vom is a linear function of the power
    slope = gen_dict["cost_curve"]["slope"]
    intercept = gen_dict["cost_curve"]["intercept"]

    m.vom = pyo.Expression(expr=slope * m.power + intercept * m.op_mode)

    return


def build_renewable_gen_operation_model(m):
    '''
    build renewable generator operation models
    '''

    return