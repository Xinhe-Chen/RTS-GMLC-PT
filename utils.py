import numpy as np
import pandas as pd
import pyomo.environ as pyo
import os
import json
import importlib.resources
import idaes.logger as idaeslog

_logger = idaeslog.getLogger(__name__)

def read_gmlc_gen(csv_path=None, type="gen_csv"):
    '''
    if the csv_path is provided, read it. Otherwise read from dispatches_sample_data
    '''
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        with importlib.resources.files('dispatches_sample_data.rts_gmlc.SourceData').joinpath(type).open('r') as f:
            df = pd.read_csv(f)

    return df


def _prescient_output_to_df(file_name):
    '''Helper for loading data from Prescient output csv.
        Combines Datetimes into single column.
    '''
    df = pd.read_csv(file_name)
    df['Datetime'] = \
        pd.to_datetime(df['Date']) + \
        pd.to_timedelta(df['Hour'], 'hour') + \
        pd.to_timedelta(df['Minute'], 'minute')
    df.drop(columns=['Date','Hour','Minute'], inplace=True)
    # put 'Datetime' in front
    cols = df.columns.tolist()
    cols = cols[-1:]+cols[:-1]
    
    return df[cols]


def make_lmp_csv(lmp_path, bus_details_path, bus_name):
    bdf = _prescient_output_to_df(bus_details_path)
    bdf = bdf[bdf["Bus"] == bus_name][["Datetime","LMP","LMP DA"]]
    bdf.set_index("Datetime", inplace=True)

    if lmp_path == None:
        _logger.info("Empty path, make a new df")
        bdf = bdf.rename(columns={'LMP': f'{bus_name}_LMP', "LMP DA": f'{bus_name}_LMP_DA'})
        lmp_df = bdf
        lmp_df.to_csv("Data/all_bus_lmp.csv")
    else:
        _logger.info(f"Extracting LMP for {bus_name}")
        bdf = bdf.rename(columns={"LMP": f"{bus_name}_LMP", "LMP DA": f"{bus_name}_LMP_DA"})
        lmp_df = pd.read_csv(lmp_path).set_index('Datetime')
        # check if the bus has already been read
        if (f'{bus_name}_LMP' in lmp_df.columns) or (f'{bus_name}_LMP_DA' in lmp_df.columns):
            _logger.info("Bus LMP already exists.")
        else:
            bdf_aligned = bdf.reindex(lmp_df.index)
            lmp_df_merge = pd.concat([lmp_df, bdf_aligned], axis=1)
            lmp_df_merge.to_csv(lmp_path)
    
    return


def _read_json():
    '''
    read the json file to get the cost curve.
    '''
    json_path = os.path.join(os.getcwd(), 'Data', 'rtsgmle_gen_linear_cost_curve.json')
    with open(json_path, 'rb') as f:
        gen_param_dict = json.load(f)
    return gen_param_dict


def sep_fossil_renew(df):
    ''''
    Separate the fossil and renewable generators in the df
    '''
    # extract the CT, CC, STEAM, WIND, PV units
    target_fossil_gen = []
    target_renew_gen = []
    for idx, row in df.iterrows():
        if row['Unit Type'] in ['CT', 'CC', 'STEAM']:
            target_fossil_gen.append(row)
        if row['Unit Type'] in ['PV', 'WIND']:
            target_renew_gen.append(row)
    df_fossil = pd.DataFrame(target_fossil_gen)
    df_renew = pd.DataFrame(target_renew_gen)

    return df_fossil, df_renew


def _read_df(df_type, renew=False):
    gen_param_dict = _read_json()

    # get the bus id and name
    bus_path = os.path.join(os.getcwd(), "..", 'rts-gmlc', "RTS-GMLC", "RTS_Data", "SourceData", "bus.csv")
    df_bus = read_gmlc_gen(csv_path=bus_path)

    result_dict = {}
    for idx, row in df_type.iterrows():
        gen_name = row['GEN UID']
        gen_dict = {}
        gen_dict['name'] = gen_name
        bus_id= row['Bus ID']
        gen_dict['bus_name'] = df_bus[df_bus["Bus ID"]==bus_id]["Bus Name"].to_list()[0]
        gen_dict['gen_type'] = row['Unit Type']
        gen_dict['max_p'] = row['PMax MW']
        gen_dict['min_p'] = row['PMin MW']
        gen_dict['ramp'] = row['Ramp Rate MW/Min']
        gen_dict['fuel_p'] = row['Fuel Price $/MMBTU']
        gen_dict['min_down_time'] = int(row['Min Down Time Hr'])
        gen_dict['min_up_time'] = int(row['Min Up Time Hr'])
        gen_dict['start_up_time_hot'] = int(row['Start Time Hot Hr'])
        gen_dict['start_up_time_warm'] = int(row['Start Time Warm Hr'])
        gen_dict['start_up_time_cold'] = int(row['Start Time Cold Hr'])
        gen_dict['start_heat_hot'] = row['Start Heat Hot MBTU']
        gen_dict['start_heat_warm'] = row['Start Heat Warm MBTU']
        gen_dict['start_heat_cold'] = row['Start Heat Cold MBTU']
        if not renew:
            gen_dict['cost_curve'] = gen_param_dict[gen_name]
        else:
            gen_dict['cost_curve'] = {'slope': 0, 'intercept': 0} # if it is a renewable generator, set the cost curve to 0

        result_dict[gen_name] = gen_dict
    
    return result_dict

def save_gen_data(df, save_opt=False):
    df_fossil, df_renew = sep_fossil_renew(df)
    fossil_dict = _read_df(df_fossil, renew=False)
    renew_dict = _read_df(df_renew, renew=True)

    # save to json files
    if save_opt:
        all_gen_dict = {}
        all_gen_dict['fossil'] = fossil_dict
        all_gen_dict['renew'] = renew_dict
        gen_dict_path = os.path.join('Data', 'gen_dict.json')
        with open(gen_dict_path, "w") as f:
            json.dump(all_gen_dict, f)

        print("Successfully saved generator parameters to json files")

    return fossil_dict, renew_dict


# check the generator types to decide its startup types
def gen_startup_cost(gen_dict):
    '''
    Get the startup costs of the generator based on the generator parameters
    
    Args:
        gen_dict: a dictionary that stores the parameter of the generator
    
    Returns:
        start_up_cost:  a dictionary that stores the costs fo the generator
    '''
    fuel_p = gen_dict['fuel_p']
    start_heat_hot = gen_dict['start_heat_hot']
    start_heat_warm = gen_dict['start_heat_warm']
    start_heat_cold = gen_dict['start_heat_cold']
    
    start_up_cost_hot = start_heat_hot*fuel_p
    start_up_cost_warm = start_heat_warm*fuel_p
    start_up_cost_cold = start_heat_cold*fuel_p
    start_up_cost = {'hot': start_up_cost_hot, 'warm': start_up_cost_warm, 'cold': start_up_cost_cold}
    
    return start_up_cost

# def capacity_limits(
#     blk,
#     op_blocks: dict,
#     uc_data,
#     set_time,
# ):
#     """
#     Appends capacity limit constraints
#     """
#     commodity = uc_data.commodity_name
#     limits = (
#         uc_data.config.op_range_lb * uc_data.config.capacity,
#         uc_data.config.capacity,
#     )
#     commodity = {t: getattr(blk, commodity) for t, blk in op_blocks.items()}

#     @blk.Constraint(set_time)
#     def capacity_low_limit_con(_, t):
#         return limits[0] * op_blocks[t].op_mode <= commodity[t]

#     @blk.Constraint(set_time)
#     def capacity_high_limit_con(_, t):
#         return commodity[t] <= limits[1] * op_blocks[t].op_mode

def ramping_limits(
    blk,
    op_blocks,
    gen_dict,
    set_time,
    commodity="power",
):
    """
    Appends ramping constraints
    """
    _ramping_var = commodity
    startup_rate = gen_dict['min_p']
    shutdown_rate = gen_dict['min_p']
    rampup_rate = gen_dict['ramp']
    rampdown_rate = gen_dict['ramp']
    ramping_var = {t: getattr(blk, _ramping_var) for t, blk in op_blocks.items()}

    @blk.Constraint(set_time)
    def ramp_up_con(_, t):
        if t == 1:
            return pyo.Constraint.Skip

        return (
            ramping_var[t] - ramping_var[t - 1]
            <= startup_rate * op_blocks[t].startup
            + rampup_rate * op_blocks[t - 1].op_mode
        )

    @blk.Constraint(set_time)
    def ramp_down_con(_, t):
        if t == 1:
            return pyo.Constraint.Skip

        return (
            ramping_var[t - 1] - ramping_var[t]
            <= shutdown_rate * op_blocks[t].shutdown
            + rampdown_rate * op_blocks[t].op_mode
        )

def capacity_limits(
    blk,
    op_blocks,
    gen_dict,
    set_time
):
    limits = {"UB": gen_dict['max_p'], "LB": gen_dict['min_p']}
    commodity = {t: getattr(blk, "power") for t, blk in op_blocks.items()}

    @blk.Constraint(set_time)
    def capacity_up_limit_con(_, t):
        return commodity[t] <= limits["UB"] * op_blocks[t].op_mode
    
    @blk.Constraint(set_time)
    def capacity_low_limit_con(_, t):
        return limits["LB"] * op_blocks[t].op_mode <= commodity[t]
     

def startup_shutdown_constraints(
    blk,
    op_blocks,
    set_time,
    gen_dict,
):
    """
    Appends startup and shutdown constraints for a given unit/process
    """
    down_time = gen_dict['min_down_time']
    up_time = gen_dict['min_up_time']
    start_up_cost = gen_startup_cost(gen_dict)
    # startup cost contains 3 keys, 'hot', 'warm' and 'cold
    # set the parameter of start_up cost
    # Attention: the generator has 3 types of startup, hot/warm/cold are ranked by the cost from lower to higher
    key_list = list(start_up_cost.keys()) # ['hot', 'warm', 'cold]
    
    # future development can using time series to the cost of startup (e.g., depending on fuel prices)
    blk.startup_type = pyo.Var(set_time, key_list, within=pyo.Binary)
    blk.startup_cost = pyo.Param(key_list, initialize = start_up_cost)
    # this is a little different. hot start = warm time - hot time, warm start = cold time - warm time. For cold startup, no need to add constraint due to the non-decreasing property.
    # Reference paper: Tight and compact MILP formulation for the thermal unit commitment problem.
    blk.startup_duration = pyo.Param(key_list, initialize = {key_list[0]: gen_dict['start_up_time_hot'], key_list[1]: gen_dict['start_up_time_warm'], key_list[2]: gen_dict['start_up_time_cold']})
    
    @blk.Constraint(set_time)
    def binary_relationship_con(_, t):
        if t == 1:
            return pyo.Constraint.Skip

        return (
            op_blocks[t].op_mode - op_blocks[t - 1].op_mode
            == op_blocks[t].startup - op_blocks[t].shutdown
        )

    @blk.Constraint(set_time)
    def minimum_up_time_con(_, t):
        if t < up_time:
            return pyo.Constraint.Skip

        return (
            sum(op_blocks[i].startup for i in range(t - up_time + 1, t + 1))
            <= op_blocks[t].op_mode
        )

    @blk.Constraint(set_time)
    def minimum_down_time_con(_, t):
        if t < down_time:
            return pyo.Constraint.Skip

        return (
            sum(op_blocks[i].shutdown for i in range(t - down_time + 1, t + 1))
            <= 1 - op_blocks[t].op_mode
        )

    @blk.Constraint(set_time)
    def startup_type_rule_hot(_, t):
        '''
        Eq 54 in Ben's paper
        '''
        if t < blk.startup_duration['warm']:
            return pyo.Constraint.Skip
        return (
            blk.startup_type[t, 'hot'] <= sum(op_blocks[t-i].shutdown for i in range(blk.startup_duration['hot'], blk.startup_duration['warm']))
        )
    
    @blk.Constraint(set_time)
    def startup_type_rule_warm(_, t):
        '''
        Eq 54 in Ben's paper
        '''
        if t < blk.startup_duration['cold']:
            return pyo.Constraint.Skip
        return (
            blk.startup_type[t, 'warm'] <= sum(op_blocks[t-i].shutdown for i in range(blk.startup_duration['warm'], blk.startup_duration['cold']))
        )
    
    @blk.Constraint(set_time)
    def startup_type_rule(_, t):
        '''
        Eq 55 in Ben's paper
        '''
        return (
            sum(blk.startup_type[t, k] for k in key_list) <= op_blocks[t].startup
        )
    
    @blk.Expression(set_time)
    def startup_cost_expr(_, t):
        '''
        Eq 56 in Ben's paper
        '''
        return (
            pyo.Expression(expr=sum(blk.startup_cost[k] * blk.startup_type[t, k] for k in key_list))   
        )