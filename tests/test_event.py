import unittest
import pandas as pd
import re
import numpy as np
import pytest
from collections import defaultdict
from pathlib import Path
from src.cegpy import EventTree
from src.cegpy.trees.event import _paths_required_for_stratification
from pydotplus.graphviz import InvocationException


class TestEventTreeAPI():
    def setup(self):
        df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')
        self.df = pd.read_excel(df_path)

    def test_required_argument_missing_fails(self):
        pytest.raises(TypeError, EventTree)

    def test_required_argument_wrong_type_fails(self):
        dataframe = 5
        pytest.raises(ValueError, EventTree, dataframe=dataframe)

    def test_incorrect_sampling_zero_fails(self):
        szp = [('edge_1'), ('edge_1', 'edge_2')]
        pytest.raises(
            ValueError,
            EventTree,
            dataframe=self.df,
            sampling_zero_paths=szp)


class TestEventTree():
    def setup(self):
        df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')

        self.df = pd.read_excel(df_path)
        self.et = EventTree(dataframe=self.df)
        self.reordered_et = EventTree(
            dataframe=self.df,
            var_order=self.df.columns[::-1]
        )
        self.node_format = re.compile('^s\\d\\d*$')

    def test_check_sampling_zero_paths_param(self) -> None:
        """Tests the function that is checking the sampling zero paths param"""
        szp = [('Medium',), ('Medium', 'High')]
        assert self.et._EventTree__check_sampling_zero_paths_param(szp) == szp
        szp = [1, 2, 3, 4]
        assert self.et._EventTree__check_sampling_zero_paths_param(szp) is None

        szp = [('path', 'to'), (123, 'something'), 'path/to']
        assert self.et._EventTree__check_sampling_zero_paths_param(szp) is None

    def test_check_sampling_zero_get_and_set(self) -> None:
        """Tests the functions that set and get the sampling zeros"""
        assert self.et.sampling_zeros is None

        szp = [('Medium',), ('Medium', 'High')]
        self.et.sampling_zeros = szp
        assert self.et.sampling_zeros == szp

    def test_order_of_columns(self) -> None:
        assert self.reordered_et.variables == list(self.df.columns[::-1])

    def test_create_node_list_from_paths(self) -> None:
        paths = defaultdict(int)
        paths[('path',)] += 1
        paths[('path', 'to')] += 1
        paths[('path', 'away')] += 1
        paths[('road',)] += 1
        paths[('road', 'to')] += 1
        paths[('road', 'away')] += 1

        # code being tested:
        node_list = self.et._EventTree__create_node_list_from_paths(paths)

        print(node_list)
        assert len(list(paths.keys())) + 1 == len(node_list)
        assert node_list[0] == 's0'
        assert node_list[-1] == 's%d' % (len(node_list) - 1)

    def test_construct_event_tree(self) -> None:
        """Tests the construction of an event tree from a set of paths,
        nodes, and """
        EXPECTED_NODE_COUNT = 45
        assert len(self.et) == EXPECTED_NODE_COUNT
        assert len(self.et.edges) == EXPECTED_NODE_COUNT - 1
        edge_counts = self.et.edge_counts

        assert len(edge_counts) == EXPECTED_NODE_COUNT - 1
        for _, count in edge_counts.items():
            assert isinstance(count, int)

    def test_get_functions_producing_expected_data(self) -> None:
        edges = list(self.et.edges)
        assert isinstance(edges, list)
        for edge in edges:
            assert isinstance(edge, tuple)
            assert len(edge) == 3
            assert isinstance(edge[0], str)
            assert isinstance(edge[1], str)
            assert isinstance(edge[2], str)

        check_list_contains_strings(list(self.et))
        check_list_contains_strings(self.et.situations)
        check_list_contains_strings(self.et.leaves)

        edge_counts = self.et.edge_counts
        print(edge_counts)
        assert isinstance(edge_counts, dict)
        for edge, count in edge_counts.items():
            assert isinstance(edge, tuple)
            for node in edge:
                assert isinstance(node, str)
            assert isinstance(count, int)

    def test_dataframe_with_numeric_values(self) -> None:
        """Ensures figure can be produced from dataframes with
        numeric values"""
        self.df["NewColumn"] = [1] * len(self.df)
        new_et = EventTree(self.df)
        try:
            new_et.create_figure("out/test_dataframe_with_numeric_values.pdf")
        except InvocationException:
            pass
        except Exception as err:
            raise AssertionError(
                "Could not create figure with numeric data"
            ) from err
        return None

    def test_node_colours(self) -> None:
        """ Ensures that all nodes in the event tree dot graph object 
        are coloured in lightgrey """
        dot_nodes = self.et.dot_event_graph.get_nodes()
        event_node_colours = [
            n.obj_dict['attributes']['fillcolor'] for n in dot_nodes
        ]
        assert len(set(event_node_colours)) == 1
        assert event_node_colours[0] == 'lightgrey'
        

class TestIntegration():
    def setup(self):
        # stratified dataset
        med_df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')
        self.med_s_z_paths = None
        self.med_df = pd.read_excel(med_df_path)
        self.med_et = EventTree(
            dataframe=self.med_df,
            sampling_zero_paths=self.med_s_z_paths
        )

        # non-stratified dataset
        fall_df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/Falls_Data.xlsx')
        self.fall_s_z_paths = None
        self.fall_df = pd.read_excel(fall_df_path)
        self.fall_et = EventTree(
            dataframe=self.fall_df,
            sampling_zero_path=self.fall_s_z_paths
        )

    def test_categories_per_variable(self) -> None:
        expected_med_cats_per_var = {
            "Classification": 2,
            "Group": 3,
            "Difficulty": 2,
            "Response": 2,
        }
        actual_med_cats_per_var = self.med_et.categories_per_variable
        assert expected_med_cats_per_var == actual_med_cats_per_var

        expected_fall_cats_per_var = {
            "HousingAssessment": 4,
            "Risk": 2,
            "Treatment": 3,
            "Fall": 2,
        }
        actual_fall_cats_per_var = self.fall_et.categories_per_variable
        assert expected_fall_cats_per_var == actual_fall_cats_per_var


def check_list_contains_strings(str_list) -> bool:
    assert isinstance(str_list, list)
    for elem in str_list:
        assert isinstance(elem, str)


class TestUsecase():
    def setup(self):
        # stratified dataset
        med_df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')
        self.med_s_z_paths = None
        self.med_df = pd.read_excel(med_df_path)
        self.med_et = EventTree(
            dataframe=self.med_df,
            sampling_zero_paths=self.med_s_z_paths
        )

        # non-stratified dataset
        fall_df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/Falls_Data.xlsx')
        self.fall_s_z_paths = None
        self.fall_df = pd.read_excel(fall_df_path)
        self.fall_et = EventTree(
            dataframe=self.fall_df
        )

    def test_fall_cats_per_var(self):
        expected_fall_cats_per_var = {
            "HousingAssessment": 4,
            "Risk": 2,
            "Treatment": 3,
            "Fall": 2,
        }
        actual_fall_cats_per_var = self.fall_et.categories_per_variable
        assert expected_fall_cats_per_var == actual_fall_cats_per_var


class TestChangingDataFrame():
    def setup(self):
        # stratified dataset
        med_df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')
        self.med_s_z_paths = None
        self.med_df = pd.read_excel(med_df_path)
        self.med_et = EventTree(
            dataframe=self.med_df,
            sampling_zero_paths=self.med_s_z_paths
        )

        # non-stratified dataset
        fall_df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/Falls_Data.xlsx')
        self.fall_s_z_paths = None
        self.fall_df = pd.read_excel(fall_df_path)
        self.fall_et = EventTree(
            dataframe=self.fall_df,
            sampling_zero_path=self.fall_s_z_paths
        )

    def test_add_empty_column(self) -> None:
        # adding empty column
        med_empty_column_df = self.med_df
        med_empty_column_df["extra"] = ""
        med_empty_column_et = EventTree(
            dataframe=med_empty_column_df
        )
        assert med_empty_column_et.adj == self.med_et.adj

        fall_empty_column_df = self.fall_df
        fall_empty_column_df["extra"] = ""
        fall_empty_column_et = EventTree(
            dataframe=fall_empty_column_df
        )
        assert fall_empty_column_et.adj == self.fall_et.adj

    def test_add_NA_column(self) -> None:
        # adding NA column
        med_add_NA_df = self.med_df
        med_add_NA_df["extra"] = np.nan
        med_add_NA_et = EventTree(
            dataframe=med_add_NA_df
        )
        assert med_add_NA_et.adj == self.med_et.adj

        fall_add_NA_df = self.fall_df
        fall_add_NA_df["extra"] = np.nan
        fall_add_NA_et = EventTree(
            dataframe=fall_add_NA_df
        )
        assert fall_add_NA_et.adj == self.fall_et.adj

    def test_add_same_column(self) -> None:
        # adding column with no more information
        med_add_same_df = self.med_df
        med_add_same_df["extra"] = "same for all"
        med_add_same_et = EventTree(
            dataframe=med_add_same_df
        )
        assert len(med_add_same_et.leaves) == len(self.med_et.leaves)

        fall_add_same_df = self.fall_df
        fall_add_same_df["extra"] = "same for all"
        fall_add_same_et = EventTree(
            dataframe=fall_add_same_df
        )
        assert len(fall_add_same_et.leaves) == len(self.fall_et.leaves)

    def test_add_same_column_int(self) -> None:
        # adding column with no more information
        med_add_same_df = self.med_df
        med_add_same_df["extra"] = 1
        med_add_same_et = EventTree(
            dataframe=med_add_same_df
        )
        try:
            med_add_same_et.create_figure("et_fig_path.pdf")
        except InvocationException:
            pass
        assert len(med_add_same_et.leaves) == len(self.med_et.leaves)

        fall_add_same_df = self.fall_df
        fall_add_same_df["extra"] = 1
        fall_add_same_et = EventTree(
            dataframe=fall_add_same_df
        )
        try:
            fall_add_same_et.create_figure("et_fig_path.pdf")
        except InvocationException:
            pass
        assert len(fall_add_same_et.leaves) == len(self.fall_et.leaves)


class TestStratification(unittest.TestCase):
    """Tests the stratification functionality of the EventTree"""

    def test_stratified(self) -> None:
        """stratified has the wrong type."""
        with pytest.raises(ValueError):
            _ = EventTree(
                dataframe=pd.DataFrame(),
                stratified=5
            )

    def test_value_error_for_not_complete_case(self):
        """When stratified and not complete_case, ValueError raised"""
        with pytest.raises(
            ValueError,
            match=(
                r"Please manually stratify the dataset"
            )
        ):
            _ = EventTree(
                dataframe=pd.DataFrame(),
                complete_case=False,
                stratified=True,
            )

    def test_produces_paths_required_for_asym_data(self):
        """For the asym.csv dataset, expected paths are produced"""
        expected_paths = [
            ("0", "1", "0", "1"),
            ("0", "1", "1", "1"),
            ("1", "0", "1", "1"),
        ]
        data = pd.read_csv("data/Asym.csv").astype(str)
        actual_paths = _paths_required_for_stratification(data)
        self.assertEqual(expected_paths, actual_paths)

    def test_event_tree_stratifies(self):
        """For the asym.csv dataset, all paths are created"""
        expected_paths = [
            ("0", "0", "0", "0"),
            ("0", "0", "0", "1"),
            ("0", "0", "1", "0"),
            ("0", "0", "1", "1"),
            ("0", "1", "0", "0"),
            ("0", "1", "0", "1"),
            ("0", "1", "1", "0"),
            ("0", "1", "1", "1"),
            ("1", "0", "0", "0"),
            ("1", "0", "0", "1"),
            ("1", "0", "1", "0"),
            ("1", "0", "1", "1"),
            ("1", "1", "0", "0"),
            ("1", "1", "0", "1"),
            ("1", "1", "1", "0"),
            ("1", "1", "1", "1"),
        ]
        data = pd.read_csv("data/Asym.csv")
        asym_et = EventTree(
            dataframe=data,
            stratified=True,
            complete_case=True
        )
        all_paths = list(asym_et._sorted_paths)
        for path in expected_paths:
            self.assertIn(path, all_paths)

    def test_warning_raised(self):
        """Warning raised if sampling zeros are not none."""
        data = pd.read_csv("data/Asym.csv")
        with self.assertLogs("cegpy", level="WARN") as log_cm:
            _ = EventTree(
                dataframe=data,
                sampling_zero_paths=[("1", "0", "1", "1")],
                stratified=True,
                complete_case=True,
            )
        self.assertIn(
            "WARNING:cegpy.event_tree:User provided sampling_zero_paths, "
            "but these are being ignored due to 'stratified' being enabled.",
            log_cm.output
        )


class TestMissingLabels():
    def setup(self):
        array = [
            np.array(["1", "N/A", "Recover"]),
            np.array(["1", "Trt1", "N/A"]),
            np.array(["2", "Struct", "Recover"]),
            np.array(["2", "Struct", "Dont Recover"]),
            np.array(["1", "Trt1", "Recover"]),
            np.array(["1", "Trt2", "Recover"]),
            np.array(["1", "Trt2", "Dont Recover"]),
            np.array(["1", "Trt1", "Dont Recover"]),
        ]

        self.df = pd.DataFrame(array)

    def test_structural_label_string(self) -> None:
        """struct_missing_label has the wrong type."""
        with pytest.raises(ValueError):
            _ = EventTree(
                dataframe=self.df,
                struct_missing_label=5
            )

    def test_structural_label_reduction(self) -> None:
        """struct_missing_label values are reduced to '<empty string>'."""
        df_et = EventTree(
                dataframe=self.df,
                struct_missing_label="Struct"
            )
        expected_df = pd.DataFrame(
            [
                np.array(["1", "N/A", "Recover"]),
                np.array(["1", "Trt1", "N/A"]),
                np.array(["2", "", "Recover"]),
                np.array(["2", "", "Dont Recover"]),
                np.array(["1", "Trt1", "Recover"]),
                np.array(["1", "Trt2", "Recover"]),
                np.array(["1", "Trt2", "Dont Recover"]),
                np.array(["1", "Trt1", "Dont Recover"]),
            ]
        )
        assert df_et.dataframe.equals(expected_df) is True

    def test_missing_label_string(self) -> None:
        """missing label not correct type."""
        with pytest.raises(ValueError):
            _ = EventTree(
                dataframe=self.df,
                missing_label=15
            )

    def test_missing_label_reduction(self) -> None:
        """missing label replaced with 'missing'."""
        df_et = EventTree(
            dataframe=self.df,
            missing_label="N/A"
        )
        expected_df = pd.DataFrame(
            [
                np.array(["1", "missing", "Recover"]),
                np.array(["1", "Trt1", "missing"]),
                np.array(["2", "Struct", "Recover"]),
                np.array(["2", "Struct", "Dont Recover"]),
                np.array(["1", "Trt1", "Recover"]),
                np.array(["1", "Trt2", "Recover"]),
                np.array(["1", "Trt2", "Dont Recover"]),
                np.array(["1", "Trt1", "Dont Recover"]),
            ]
        )
        assert df_et.dataframe.equals(expected_df) is True

    def test_struct_missing_label_reduction(self) -> None:
        """missing_label and struct_missing_label both set."""
        df_et = EventTree(
            dataframe=self.df,
            struct_missing_label="Struct",
            missing_label="N/A"
        )
        expected_df = pd.DataFrame(
            [
                np.array(["1", "missing", "Recover"]),
                np.array(["1", "Trt1", "missing"]),
                np.array(["2", "", "Recover"]),
                np.array(["2", "", "Dont Recover"]),
                np.array(["1", "Trt1", "Recover"]),
                np.array(["1", "Trt2", "Recover"]),
                np.array(["1", "Trt2", "Dont Recover"]),
                np.array(["1", "Trt1", "Dont Recover"]),
            ]
        )
        assert df_et.dataframe.equals(expected_df) is True

    def test_complete_case_bool(self) -> None:
        """complete case wrong type"""
        with pytest.raises(ValueError):
            _ = EventTree(
                dataframe=self.df,
                complete_case="Yes"
            )

    def test_complete_case_missing_reduction(self) -> None:
        """complete case missing_label reduces"""
        df_et = EventTree(
            dataframe=self.df,
            missing_label="N/A",
            complete_case=True
        )
        expected_df = pd.DataFrame(
            [
                np.array(["2", "Struct", "Recover"]),
                np.array(["2", "Struct", "Dont Recover"]),
                np.array(["1", "Trt1", "Recover"]),
                np.array(["1", "Trt2", "Recover"]),
                np.array(["1", "Trt2", "Dont Recover"]),
                np.array(["1", "Trt1", "Dont Recover"]),
            ]
        )
        assert df_et.dataframe.equals(expected_df) is True

    def test_complete_case_reduction(self) -> None:
        """struct_missing_label reduces."""
        df_et = EventTree(
                dataframe=self.df,
                struct_missing_label="Struct",
                missing_label="N/A",
                complete_case=True
            )
        expected_df = pd.DataFrame(
            [
                np.array(["2", "", "Recover"]),
                np.array(["2", "", "Dont Recover"]),
                np.array(["1", "Trt1", "Recover"]),
                np.array(["1", "Trt2", "Recover"]),
                np.array(["1", "Trt2", "Dont Recover"]),
                np.array(["1", "Trt1", "Dont Recover"]),
            ]
        )
        assert df_et.dataframe.equals(expected_df) is True
