from ast import Str
from collections import defaultdict
import itertools
from typing import Dict, List, Mapping, Optional, Tuple
import numpy as np
from pandas.core.frame import DataFrame
import pydotplus as pdp
import logging
from ..utilities.util import Util
from IPython.display import Image
from IPython import get_ipython
import pandas as pd
import textwrap
import networkx as nx
from ..trees.event import EventTree
from pandas.api.types import is_numeric_dtype
# create logger object for this module
logger = logging.getLogger('cegpy.staged_tree')


class CTTree(EventTree):
    """
    Class for continuous time event trees. Inherits from the event
    tree class.

    Additional Parameters
    ----------

     holding_time_columns: A mapping of variable column names to its corresponding
                holding time column names, to be passed as a dict:
                holding_time_columns={
                    "first symptom column": "time to first symptom column",
                    "second symptom column": "time to second symptom column",
                    ...
                }.
            All holding times must be integers or floats and must be in the
            same unit of time (e.g. seconds, minutes, hours, days). 
            Any missing values in the holding times must be saved as np.nan.
    """


    def __init__(
            self,
            dataframe: pd.DataFrame,
            sampling_zero_paths=None,
            incoming_graph_data=None,
            var_order=None,
            struct_missing_label=None,
            missing_label=None,
            complete_case=False,
            stratified=False,
            holding_time_columns=None,
            **attr
            ) -> None:

        # Checking argument inputs are sensible
        if holding_time_columns is None:
            raise ValueError(
                "Perhaps you should use the EventTree class"
            )

        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError(
                "The dataframe parameter must be a pandas.DataFrame"
            )
            
        
        self.time_columns = list(
            holding_time_columns.values()
        )

        # First set of checks that holding_time_columns input is sensible 
        self._holding_time_checks_one(
            dataframe, 
            holding_time_columns
        )

        new_dataframe = dataframe.drop(
            columns = list(holding_time_columns.values()),
            inplace = False,
        )

        # Call event tree init to generate event tree
        super().__init__(
            new_dataframe,
            sampling_zero_paths,
            incoming_graph_data,
            var_order,
            struct_missing_label,
            missing_label,
            complete_case,
            stratified,
            **attr
        )

        if (missing_label is not None and
            complete_case is True):


        # Second set of checks that holding_time_columns input is sensible 
        self._holding_time_checks_two(
            new_dataframe, 
            dataframe,
            holding_time_columns
        )


    def _holding_time_checks_one(
        self, 
        dataframe, 
        holding_time_columns
    ) -> None:
        # 1) check that the keys and values are column names
        all_columns = list(dataframe.columns)
        ct_columns = list(holding_time_columns.keys())
        leftover_ct_columns = [
            col for col in ct_columns if col not in all_columns
            ]
        
        if not leftover_ct_columns:
            raise ValueError(
                f"Columns {leftover_ct_columns} are not in the dataframe"
            )
        
        leftover_time_columns = [
            col for col in self.time_columns\
                 if col not in all_columns
        ]
        if not leftover_time_columns:
            raise ValueError(
                f"Columns {leftover_time_columns} are not in the dataframe"
            )

        # 2) check that all values in time_columns are numeric/float...
        # ... and check that all values are non-negative
        for col in self.time_columns:
            if not is_numeric_dtype(dataframe[col]):
                raise ValueError(
                    f"Column {col} does not have numeric values \n"
                    "You could coerce these into numeric using pd.to_numeric()"
                )
            if (dataframe[col].values < 0).any():
                raise ValueError(
                    f"Column {col} has negative entries"
                )


    def _holding_time_checks_two(
        self, 
        new_dataframe, 
        dataframe,
        holding_time_columns
    ) -> None:
        # 3) check that all rows that are observed in ct_columns ...
        # ... are also observed in time_columns
        ct_columns = list(holding_time_columns.keys())
        for ind, col in enumerate(ct_columns):
            struct_values = new_dataframe.index[
                new_dataframe[col] == ""
            ].tolist()
            miss_values = new_dataframe.index[
                new_dataframe[col] == "missing"
            ].tolist()
            total_miss = struct_values + miss_values
            time_col = self.time_columns[ind]
            time_miss_values = dataframe.index[
                dataframe[time_col] == np.nan
            ].tolist()
            
            problem_indices = [
                ind for ind in time_miss_values if ind\
                    not in total_miss
            ]

            if problem_indices:
                raise ValueError(
                    f"Current functionality of the package does not support\
                        the case where a holding time observation is missing\
                            if its corresponding variable value has been observed."
                )



            
        




        