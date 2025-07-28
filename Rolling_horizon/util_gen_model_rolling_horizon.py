import numpy as np
import pandas as pd
import pyomo.environ as pyo
import idaes.logger as idaeslog

_logger = idaeslog.getLogger(__name__)

# def build_fossil_gen_design_model(m, params):
#     '''
#     build generator design models. 
#     We do not do design optimization in this work so the maximun capacity is a parameter.  
    
#     Args:
#         m: pyomo model.
#         params: dictionary that stores the generator parameters.

#     Returns:
#         None
#     '''
#     # if fom and capex not provide, just set them to 0
#     for key in params.keys():
#         if "capex" not in list(params[key].keys()):
#             params[key]["capex"] = 0

#         if "fom" not in list(params[key].keys()):
#             params[key]["fom"] = 0

#         setattr(getattr(m, f"gen_design_{key}"), 
#                 "capacity", 
#                 pyo.Param(
#                     initialize=params[key]['max_p'],
#                     mutable=True,
#                     doc="Maxium capacity of the generator [in MW]",
#                 )
#             )
        
#         m.capex = pyo.Expression(
#             expr=getattr(getattr(m, f"gen_design_{key}"), "capacity") * params[key]["capex"],
#             )
        
#         m.fom = pyo.Expression(
#             expr=getattr(getattr(m, f"gen_design_{key}"), "capacity") * params[key]["fom"]
#         )
    
#     return


# def build_fossil_gen_operation_model(m, params):
#     """
#     Function that adds the fossil generator operation model

#     Args:
#         m: Pyomo Block
#         design_blk: Pyomo block containing the design model
#         params: dictionary that stores the generator parameters.

#     Returns:
#         None
#     """
#     # the power output at each time period
#     for key in params.keys():
#         setattr(getattr(m, f"gen_design_{key}"), 
#                 "power", 
#                 pyo.Var(
#                     within=pyo.NonNegativeReals,
#                     bounds=(0, params['max_p']),
#                     doc="Net power produced by NGCC at time t [in MW]",
#                 )
#             )        

#         # vom is a linear function of the power
#         slope = params[key]["cost_curve"]["slope"]
#         intercept = params[key]["cost_curve"]["intercept"]

#         power = getattr(getattr(m, f"gen_design_{key}"), "power")
#         op_mode = getattr(getattr(m, f"gen_design_{key}"), "op_mode")
#         startup = getattr(getattr(m, f"gen_design_{key}"), "startup")
#         shutdown = getattr(getattr(m, f"gen_design_{key}"), "shutdown")

#         m.vom = pyo.Expression(expr=slope * power + intercept * op_mode)

#         m.startup_cost = pyo.Expression(expr=params[key]["fuel_p"]*params[key]["start_heat_cold"]*startup)
#         m.shutdown_cost = pyo.Expression(expr=params[key]["fuel_p"]*0.0*shutdown)

#     return


def build_fossil_gen_design_model(m, params):
    '''
    build generator design models. 
    We do not do design optimization in this work so the maximun capacity is a parameter.  
    
    Args:
        m: pyomo model.
        params: dictionary that stores the generator parameters.

    Returns:
        None
    '''
    # if fom and capex not provide, just set them to 0
    if "capex" not in list(params.keys()):
        params["capex"] = 0

    if "fom" not in list(params.keys()):
        params["fom"] = 0

    m.capacity = pyo.Param(
        initialize=params['max_p'],
        mutable=True,
        doc="Maxium capacity of the generator [in MW]",
    )
    m.capex = pyo.Expression(
        expr=m.capacity * params["capex"],
        )
    
    m.fom = pyo.Expression(
        expr=m.capacity * params["fom"]
    )
    
    return


def build_fossil_gen_operation_model(m, params):
    """
    Function that adds the fossil generator operation model

    Args:
        m: Pyomo Block
        params: dictionary that stores the generator parameters.

    Returns:
        None
    """
    # the power output at each time period
    m.power = pyo.Var(
        within=pyo.NonNegativeReals,
        doc="Net power produced by NGCC at time t [in MW]",
        bounds=(0, params['max_p']),
        # doc="Output of the power at time t"
    )
    # placeholder: theorically, we can calculate the CO2 emission.

    # vom is a linear function of the power
    slope = params["cost_curve"]["slope"]
    intercept = params["cost_curve"]["intercept"]

    m.vom = pyo.Expression(expr=slope * m.power + intercept * m.op_mode)

    m.startup_cost = pyo.Expression(expr=params["fuel_p"]*params["start_heat_cold"]*m.startup)
    m.shutdown_cost = pyo.Expression(expr=params["fuel_p"]*0.0*m.shutdown)

    return