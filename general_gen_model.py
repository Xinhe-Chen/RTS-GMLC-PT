import re
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import idaes.logger as idaeslog
from idaes.core.util.config import ConfigurationError
from pyomo.environ import (
    ConcreteModel,
    Block,
    Var,
    Param,
    RangeSet,
    Objective,
    Constraint,
    NonNegativeReals,
    Expression,
    maximize,
    # creating a new name to avoid pylint warning on outerscope variable
    value as pyo_value,
)
# from util_gen_model import build_renewable_gen_model, build_fossil_gen_model
from Unit_Commitment import capacity_limits, ramping_limits, startup_shutdown_constraints

_logger = idaeslog.getLogger(__name__)


'''
This file is adapted from https://github.com/IDAES/idaes-pse/tree/main/idaes/apps/grid_integration/pricetaker
This file is mainly to run the price-taker model with changeable startup costs for fossil generators in RTS-GMLC
The main changes in this file include:
    1. Remove day sets
    2. Add changeable ramping limits
'''

class GmlcGen():
    '''
    This is a class to build Price-taker (PT) model 
    of generator models in RTS-GMLC dataset.
    This class needs to be generalized in the PriceTakerRTSGMLC.
    '''
    def __init__(self, bus_id, gen_dict, lmp_path):
        self.bus_id = bus_id    # use bus id to search LMP signals.
        self.gen_dict = gen_dict    # all the generator parameters goes here.
        self.lmp_path = lmp_path    # lmp path of the csv_file.

    @property
    def bus_id(self):
        '''Returns the bus id'''
        return self._bus_id
    
    @bus_id.setter
    def bus_id(self, value):
        '''Setter for the bus_id property'''
        self._bus_id = value
    
    @property
    def gen_dict(self):
        '''Returns the gen_dict'''
        return self._gen_dict
    
    @gen_dict.setter
    def bus_id(self, value):
        '''Setter for the gen_dict property'''
        self._gen_dict = value

    def _check(self):
        '''
        Check the value of the input parameters
        '''
        if not isinstance(self.gen_dict["bus_id"], str, int):
            raise TypeError(f"bus_id must be an int or str, but {type(self.bus_id)} is provided.")
        
        '''
        check the keys in the self.gen, 
        make sure all the generator params are provided and correct
        '''
        wanted_keys = ["max_p", "min_p", "ramp_rate", "min_down_time", "min_up_time"]
        keys = self.gen_dict.keys()
        for k in wanted_keys:
            if k not in keys:
                raise KeyError(f"{k} must be provided for in the gen_dict.")
        
        return


class PriceTakerRTSGMLC(ConcreteModel):
    '''
    A simplified price-taker class comparing to idaes/apps/grid_integration/pricetaker/price_taker_model.py
    '''
    def __init__(self, gen_dict, lmp_path, *args, **kwds):
        super().__init__(*args, **kwds)
        self.gen_dict = gen_dict
        self.lmp_path = lmp_path
        self._has_hourly_cashflows = False
        self._has_overall_cashflows = False
        self._linking_constraint_counter = 1


    def _read_lmp(self):
        '''
        Read the csv file to get lmp.
        '''
        df_lmp = pd.read_csv(self.lmp_path)
        lmp_data = df_lmp[self.gen_dict["bus_name"]+"_LMP"]

        return lmp_data
    

    def _assert_mp_model_exists(self):
        """
        Raise an error if the multiperiod model does not exist
        """
        if not hasattr(self, "period"):
            raise ConfigurationError(
                "Unable to find the multiperiod model. Please use the "
                "build_multiperiod_model method to construct one."
            )
        

    def build_multiperiod_model(
        self,
        flowsheet_func,
        flowsheet_options
    ):
        '''
        Build the price-taker multiperiod model (use some code from Pricetaker Class)
        
        Args:
        flowsheet_func : Callable
            A function that returns an instance of the flowsheet

        flowsheet_options : dict,
            Optional arguments needed for `flowsheet_func`
        '''
        lmp_data = self._read_lmp()

        if hasattr(self, "period"):
            # Object may contain a multiperiod model. so raise an error
            raise ConfigurationError(
                "A multiperiod model might already exist, as the object has "
                "`period` attribute."
            )
        
        # the length of the PT denpends on the LMP length
        self.horizon_length = len(lmp_data)
        self.set_time = RangeSet(self.horizon_length)

        flowsheet_blk = ConcreteModel()
        if flowsheet_options is None:
            flowsheet_options = {}
        flowsheet_func(flowsheet_blk, **flowsheet_options)

        # here the perid is 1D instead of 2D. 
        self.period = Block(self.set_time)
        for t, i in zip(self.period, range(len(lmp_data))):
            self.period[t].transfer_attributes_from(flowsheet_blk.clone())

            # If the flowsheet model has LMP defined, update it.
            if hasattr(self.period[t], "LMP"):
                self.period[t].LMP = lmp_data[i]

            # Iterate through model to append LMP data if it's been defined
            for blk in self.period[t].component_data_objects(Block):
                if hasattr(blk, "LMP"):
                    blk.LMP = lmp_data[i]


    def _get_operation_blocks(
        self,
        blk_name: str,
        attribute_list: list,
    ):
        """
        Returns a dictionary of operational blocks named 'blk_name'.
        In addition, it also checks the existence of the operational
        blocks, and the existence of specified attributes.
        """
        # Ensure that the multiperiod model exists
        self._assert_mp_model_exists()

        # pylint: disable=not-an-iterable
        op_blocks = {t: self.period[t].find_component(blk_name) for t in self.set_time}

        # NOTE: It is sufficient to perform checks only for one block, because
        # the rest of them are clones.
        blk = op_blocks[1]  # This object always exists

        # First, check for the existence of the operational block
        if blk is None:
            raise AttributeError(f"Operational block {blk_name} does not exist.")

        # Next, check for the existence of attributes.
        for attribute in attribute_list:
            if not hasattr(blk, attribute):
                raise AttributeError(
                    f"Required attribute {attribute} is not found in "
                    f"the operational block {blk_name}."
                )

        return op_blocks
    

    def add_linking_constraints(self, previous_time_var: str, current_time_var: str):
        """
        Adds constraints to relate variables across two consecutive time periods. This method is
        usually needed if the system has storage. Using this method, the holdup at the end of the
        previous time period can be equated to the holdup at the beginning of the current time
        period.

        Args:
            previous_time_var : str,
                Name of the operational variable at the end of the previous time step

        current_time_var : str,
                Name of the operational variable at the beginning of the current time step

        """
        old_time_var = self._get_operation_vars(previous_time_var)
        new_time_var = self._get_operation_vars(current_time_var)

        def _rule_linking_constraints(_, t):
            if t == 1:
                return Constraint.Skip
            return old_time_var[t - 1] == new_time_var[t]

        setattr(
            self,
            "variable_linking_constraints_" + str(self._linking_constraint_counter),
            Constraint(self.set_days, self.set_time, rule=_rule_linking_constraints),
        )
        _logger.info(
            f"Linking constraints are added to the model at "
            f"variable_linking_constraints_{self._linking_constraint_counter}"
        )
        self._linking_constraint_counter += 1  # Increase the counter for the new pair


    @staticmethod
    def _get_valid_block_name(blk_name: str):
        "Returns a valid python variable name for the given block"
        # Indexed blocks contain square brackets. This method replaces
        # them with underscores and returns a valid python variable name.
        bn = re.sub(r"[\[,\]]", "_", blk_name.split(".")[-1])
        if bn[-1] == "_":
            # Remove the trailing underscore, if it exists
            return bn[:-1]
        return bn
    

    def add_capacity_limits(
        self,
        op_block_name,
        commodity = "power",
    ):
        # _get_operation_blocks method ensures that the operation block exists, and
        # the commodity variable also exists.
        op_blocks = self._get_operation_blocks(
            blk_name=op_block_name, attribute_list=["op_mode", commodity]
        )

        # Create a block for storing capacity limit constraints
        cap_limit_blk_name = (
            self._get_valid_block_name(op_block_name) + f"_{commodity}_limits"
        )
        if hasattr(self, cap_limit_blk_name):
            raise ConfigurationError(
                f"Attempting to overwrite capacity limits for {commodity} in {op_block_name}."
            )

        setattr(self, cap_limit_blk_name, Block())
        cap_limit_blk = getattr(self, cap_limit_blk_name)

        # pylint: disable = not-an-iterable
        capacity_limits(
                blk=cap_limit_blk,
                op_blocks=op_blocks,
                gen_dict=self.gen_dict,
                set_time=self.set_time,
        )

        # Logger info for where constraint is located on the model
        _logger.info(
            f"Created capacity limit constraints for commodity {commodity} in "
            f"operation block {op_block_name} at {cap_limit_blk.name}"
        )

    def add_ramping_limits(
        self,
        op_block_name,
        commodity="power",
    ):
        """
        Adds ramping constraints of the form:
        ramping_var[t] - ramping_var[t-1] <=
        startup_rate * capacity * startup[t] + rampup_rate * capacity * op_mode[t-1];
        ramping_var[t-1] - ramping_var[t] <=
        shutdown_rate * capacity * shutdown[t] + rampdown_rate * capacity * op_mode[t]

        Args:
            op_block_name: str,
                Name of the operation model block, e.g., ("fs.ngcc")

            commodity: str,
                Name of the variable that the ramping constraints will be applied to,
                e.g., "power"
        """
        # Get operational blocks
        op_blocks = self._get_operation_blocks(
            blk_name=op_block_name,
            attribute_list=["op_mode", "startup", "shutdown", commodity],
        )
        # Creating the pyomo block
        ramp_blk_name = (
            self._get_valid_block_name(op_block_name) + f"_{commodity}_ramping"
        )
        if hasattr(self, ramp_blk_name):
            raise ConfigurationError(
                f"Attempting to overwrite ramping limits for {commodity} in {op_block_name}."
            )

        setattr(self, ramp_blk_name, Block())
        ramp_blk = getattr(self, ramp_blk_name)
    
        ramping_limits(
                blk=ramp_blk,
                op_blocks=op_blocks,
                gen_dict=self.gen_dict,
                set_time=self.set_time,
                )

        # Logger info for where constraint is located on the model
        _logger.info(
            f"Created ramping constraints for variable {commodity} "
            f"on operational block {op_block_name} at {ramp_blk.name}"
        )

    
    def add_startup_shutdown(
        self,
        op_block_name: str,
    ):
        """
        Adds minimum uptime/downtime constraints for a given unit/process

        Args:
            op_block_name: str,
                Name of the operation model block, e.g., "fs.ngcc"

            des_block_name: str, default=None,
                Name of the design model block for the operation block
                op_block_name. This argument is specified if the design is
                being optimized simultaneously, e.g., "ngcc_design"
        """
        op_blocks = self._get_operation_blocks(
            blk_name=op_block_name,
            attribute_list=["op_mode", "startup", "shutdown"],
        )

        start_shut_blk_name = (
            self._get_valid_block_name(op_block_name) + "_startup_shutdown"
        )
        if hasattr(self, start_shut_blk_name):
            raise ConfigurationError(
                f"Attempting to overwrite startup/shutdown constraints "
                f"for operation block {op_block_name}."
            )

        setattr(self, start_shut_blk_name, Block())
        start_shut_blk = getattr(self, start_shut_blk_name)

        # pylint: disable=not-an-iterable
        startup_shutdown_constraints(
                blk=start_shut_blk,
                op_blocks=op_blocks,
                set_time=self.set_time,
                gen_dict=self.gen_dict,
            )

        # Logger info for where constraint is located on the model
        _logger.info(
            f"Created startup/shutdown constraints for operation model "
            f" {op_block_name} at {start_shut_blk.name}."
        )

    def add_hourly_cashflows(
        self,
        op_block_name,
        revenue_streams,
        operational_costs,
    ):
        """
        Add hourly revenue for each operation block
        """
        self._assert_mp_model_exists()

        if operational_costs is None:
            _logger.warning(
                "Argument operational_costs is not specified, so the total "
                "operational cost will be set to 0."
            )
            operational_costs = []

        if revenue_streams is None:
            _logger.warning(
                "Argument revenue_streams is not specified, so the total "
                "revenue will be set to 0."
            )
            revenue_streams = []


        # Set net profit contribution expressions to 0
        # Here, the period is indexed by t

        for t in self.set_time:
            hourly_cost_expr = 0
            hourly_revenue_expr = 0
            blk = self.period[t]
            for cost in operational_costs:
                curr_cost = getattr(blk, "gen_"+self.gen_dict["name"]).find_component(cost)
                hourly_cost_expr += curr_cost if curr_cost is not None else 0

            # Add revenue streams for each block. If more than one block, may have
            # revenues that exist on one block and not on another. (i.e., coproduction)
            for revenue in revenue_streams:
                curr_rev = blk.find_component(revenue)
                hourly_revenue_expr += curr_rev if curr_rev is not None else 0


            # Set total cost expression
            self.period[t].total_hourly_cost = Expression(expr=hourly_cost_expr)

            # Set total revenue expression
            self.period[t].total_hourly_revenue = Expression(expr=hourly_revenue_expr)

            # Set the startup cost
            start_shut_blk_name = (
            self._get_valid_block_name(op_block_name) + "_startup_shutdown"
            )
            start_shut_blk = getattr(self, start_shut_blk_name)
            self.period[t].hourly_startup_cost = pyo.Expression(expr=start_shut_blk.startup_cost_at_t[t])

            # Set net cash inflow expression
            self.period[t].net_hourly_cash_inflow = Expression(
                expr=self.period[t].total_hourly_revenue
                - self.period[t].total_hourly_cost
                - self.period[t].hourly_startup_cost
            )

        # Logger info for where constraint is located on the model
        self._has_hourly_cashflows = True

    
    def add_overall_cashflows(
        self,
        lifetime=30,
        discount_rate=0.08,
        corporate_tax_rate=0.2,
        annualization_factor=None,
        ):
        
        # Ensure that multiperiod model exists
        self._assert_mp_model_exists()
        # The goal here is to optimize the annual profit
        # no capex calculation required
        if not self._has_hourly_cashflows:
            raise ConfigurationError(
                "Hourly cashflows are not added to the model. Please run "
                "add_hourly_cashflows method before calling the "
                "add_overall_cashflows method."
            )

        self.cashflows = Block()
        self.cashflows.capex = Var(within=NonNegativeReals, doc="Total CAPEX")
        self.cashflows.capex_calculation = Constraint(expr=self.cashflows.capex == 0)

        self.cashflows.fom = Var(within=NonNegativeReals, doc="Yearly Fixed O&M Cost")
        self.cashflows.fom_calculation = Constraint(expr=self.cashflows.fom == 0)

        self.cashflows.depreciation = Var(within=NonNegativeReals, doc="Yearly depreciation")
        self.cashflows.depreciation_calculation = Constraint(
            expr=self.cashflows.depreciation == self.cashflows.capex / lifetime
        )

        self.cashflows.net_cash_inflow = Var(doc="Net cash inflow")

        self.cashflows.net_cash_inflow_calculation = Constraint(
            expr=(
                self.cashflows.net_cash_inflow
                == sum(
                    self.period[t].net_hourly_cash_inflow
                    for t in self.period
                )
            )
        )

        self.cashflows.corporate_tax = Var(within=NonNegativeReals, doc="Corporate tax")
        self.cashflows.corporate_tax_calculation = Constraint(
            expr=self.cashflows.corporate_tax
            >= corporate_tax_rate * (self.cashflows.net_cash_inflow - self.cashflows.fom - self.cashflows.depreciation)
        )

        self.cashflows.net_profit = Var(doc="Net profit after taxes")
        self.cashflows.net_profit_calculation = Constraint(
            expr=self.cashflows.net_profit == self.cashflows.net_cash_inflow - self.cashflows.fom - self.cashflows.corporate_tax
        )

        if annualization_factor is None:
            # If the annualization factor is not specified
            annualization_factor = discount_rate / (
                1 - (1 + discount_rate) ** (-lifetime)
            )

        self.cashflows.lifetime_npv = Expression(
            expr=(1 / annualization_factor) * self.cashflows.net_profit - self.cashflows.capex
        )
        self.cashflows.npv = Expression(
            expr=self.cashflows.net_profit - annualization_factor * self.cashflows.capex,
        )

        self._has_overall_cashflows = True
        _logger.info(f"Overall cashflows are added to the block {self.cashflows.name}")

    
    def add_objective_function(self, objective_type="npv"):
        # pylint: disable = attribute-defined-outside-init
        if not self._has_overall_cashflows:
            raise ConfigurationError(
                "Overall cashflows are not appended. Please run the "
                "add_overall_cashflows method."
            )

        try:
            self.obj = Objective(
                expr=getattr(self.cashflows, objective_type),
                sense=maximize,
            )

        except AttributeError as msg:
            raise ConfigurationError(
                f"{objective_type} is not a supported objective function."
                f"Please specify either npv, or lifetime_npv, or net_profit "
                f"as the objective_type."
            ) from msg


    def _get_num_startups(self, op_block_name):
        """
        Get the number of startups in the result
        """
        op_blks = self._get_operation_blocks(
            blk_name=op_block_name, attribute_list=["startup"]
        )

        startups = {t: pyo.value(op_blks[t].startup) for t in self.period}

        try:
            total_startups = sum(startups[t] for t in self.period)
            return total_startups
        
        except TypeError:
            # Either the problem is not solved, or the startup variable is not used
            _logger.warning(
                "startup variable value is not available. \n\t Either the model "
                "is not solved, or the startup variable maynot be used in the model."
            )
            return None
        
    
    def _get_num_shutdowns(self, op_block_name):
        """Returns the number of times the given operation block undergoes shutdown"""
        op_blks = self._get_operation_blocks(
            blk_name=op_block_name, attribute_list=["shutdown"]
        )
        shutdowns = {t: pyo.value(op_blks[t].shutdown) for t in self.period}

        # pylint: disable = not-an-iterable
        try:
            total_shutdowns = sum(shutdowns[t] for t in self.period)
            return total_shutdowns
        
        except TypeError:
            # Either the problem is not solved, or the shutdown variable is not used
            _logger.warning(
                "startup variable value is not available. \n\t Either the model "
                "is not solved, or the startup variable maynot be used in the model."
            )
            return None
    

    def _get_operation_var_values(self, var_list=None):
        """
        Returns a DataFrame of all operation values

        Args:
            var_list : list, optional
                List of variables/expressions. If not specified, values of all variables
                and expressions will be returned, by default None
        """
        result = {
            "Time": [t for t in self.period],
            "LMP": [pyo.value(getattr(self.period[t], "gen_"+self.gen_dict["name"]).LMP) for t in self.period]
        }

        if var_list is not None:
            # Return the values of selected variables and/or expressions
            for v in var_list:
                result[v] = [pyo_value(self.period.find_component(v))]

        else:
            # Return the values of all variables and expressions
            blk = self.period[1]  # Reference block to extract variable names
            for v in blk.component_data_objects(Var):
                # Variable name will be of the form period[d, t].var_name
                v_name = v.name.split(".", maxsplit=1)[-1]
                result[v_name] = []
                for t in self.period:
                    result[v_name].append(pyo.value(self.period[t].find_component(v_name)))

            for v in blk.component_data_objects(Expression):
                # Expression name will be of the form period[d, t].expr_name
                v_name = v.name.split(".", maxsplit=1)[-1]
                result[v_name] = []
                for t in self.period:
                    result[v_name].append(pyo_value(self.period[t].find_component(v_name)))

        # Return the data as a DataFrame
        return pd.DataFrame(result)


    def get_results(self, op_block_name, var_list=None):
        startups = self._get_num_startups(op_block_name)
        shutdowns = self._get_num_shutdowns(op_block_name)

        _logger.info(f"startups: {startups}")
        _logger.info(f"shutdowns: {shutdowns}")

        df_result = self._get_operation_var_values()
        
        return df_result