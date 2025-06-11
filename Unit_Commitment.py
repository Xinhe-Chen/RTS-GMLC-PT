import pyomo.environ as pyo
from utils import gen_startup_cost

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
    blk.startup_cost_at_t = pyo.Var(set_time, initialize = {t:0 for t in set_time} )

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
            sum(blk.startup_type[t, k] for k in key_list) == op_blocks[t].startup
        )
    
    @blk.Constraint(set_time)
    def startup_cost_expr(_, t):
        '''
        Eq 56 in Ben's paper
        '''
        return (
            sum(blk.startup_cost[k] * blk.startup_type[t, k] for k in key_list) == blk.startup_cost_at_t[t]
        )