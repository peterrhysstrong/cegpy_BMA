from copy import deepcopy
from fractions import Fraction
from operator import add, sub, itemgetter
from IPython.display import Image
from itertools import combinations, chain
from typing import List, Optional, Tuple, Union
import networkx as nx
import scipy.special
import logging
from ..utilities.util import Util
from ..trees.event import EventTree
import random
import numpy as np
import random 

logger = logging.getLogger('cegpy.staged_tree')


class StagedTree(EventTree):

    _edge_attributes: List = [
        'count', 
        'prior', 
        'posterior',
        'probability' 
        ]

    def __init__(
            self,
            dataframe,
            sampling_zero_paths=None,
            incoming_graph_data=None,
            var_order=None,
            struct_missing_label=None,
            missing_label=None,
            complete_case=False,
            stratified=False,
            **attr) -> None:

        # Call event tree init to generate event tree
        super().__init__(
            dataframe=dataframe,
            sampling_zero_paths=sampling_zero_paths,
            incoming_graph_data=incoming_graph_data,
            var_order=var_order,
            struct_missing_label=struct_missing_label,
            missing_label=missing_label,
            complete_case=complete_case,
            stratified=stratified,
            **attr
        )

        self._mean_posterior_probs = []
        self._merged_situations = []
        self._stage_colours = []
        self._sort_count = 0
        self._colours_for_situations = []
        logger.debug("Starting Staged Tree")

    @property
    def prior(self):
        return nx.get_edge_attributes(self, 'prior')

    @prior.setter
    def prior(self, prior):
        offset = 0
        for node_idx, node_priors in enumerate(prior):
            node_name = ('s%d' % (node_idx + offset))
            while self.succ[node_name] == {}:
                offset += 1
                node_name = ('s%d' % (node_idx + offset))
            pass
            for edge_prior_idx, succ_key in enumerate(
                    self.succ[node_name].keys()):
                label = list(self.succ[node_name][succ_key])[0]
                self.edges[(node_name, succ_key, label)]['prior'] = \
                    node_priors[edge_prior_idx]

    @property
    def prior_list(self):
        """Priors provided as a list of lists"""
        prior_list = []
        prev_node = list(self.prior)[0][0]
        succ_list = []

        for edge, prior in self.prior.items():
            node = edge[0]
            if node != prev_node:
                prior_list.append(succ_list)
                succ_list = []
            succ_list.append(prior)
            prev_node = node

        if succ_list != []:
            prior_list.append(succ_list)
        return prior_list

    @property
    def posterior(self):
        '''Posterior is calculated such that the edge count is added
        to the prior for each edge.'''
        try:
            posterior = nx.get_edge_attributes(self, 'posterior')
            if posterior == {}:
                raise AttributeError('Posterior not yet set.')
            else:
                return posterior
        except AttributeError:
            for edge in self.edges:
                edge_dict = self.edges[edge]
                posterior = edge_dict['prior'] + edge_dict['count']
                edge_dict['posterior'] = posterior
            return nx.get_edge_attributes(self, 'posterior')

    @property
    def posterior_list(self):
        posterior_list = []
        prev_node = list(self.posterior)[0][0]
        succ_list = []

        for edge, posterior in self.posterior.items():
            node = edge[0]
            if node != prev_node:
                posterior_list.append(succ_list)
                succ_list = []
            succ_list.append(posterior)
            prev_node = node

        if succ_list != []:
            posterior_list.append(succ_list)
        return posterior_list

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def hyperstage(self):
        return self._hyperstage

    @hyperstage.setter
    def hyperstage(self, value):
        self._hyperstage = value

    @property
    def edge_countset(self):
        return self._create_edge_countset()

    @property
    def ahc_output(self):
        return self._ahc_output

    @ahc_output.setter
    def ahc_output(self, value):
        self._ahc_output = value

    def _check_hyperstages(self, hyperstage) -> None:
        hyper_situations = chain(*hyperstage)
        hyper_situations_set = set(hyper_situations)
        situations_set = set(self.situations)
        # Check if all situations are present
        missing = situations_set.difference(hyper_situations_set)
        if missing:
            raise ValueError(
                f"Situation(s) {missing} are missing from the list of "
                "hyperstages."
            )
        # Check if all situations provided exist
        extra = hyper_situations_set.difference(situations_set)
        if extra:
            raise ValueError(
                f"Situation(s) {extra} are not present in the tree."
            )
        # Check if all situations in a stage have the same number of edges
        for stage in hyperstage:
            n_edges = self.out_degree[stage[0]]
            for node in stage:
                if self.out_degree[node] != n_edges:
                    raise ValueError(
                        "Situations in the same hyperstage "
                        "must have the same number of outgoing edges."
                    )

    def _check_prior(self, prior) -> None:
        if len(prior) != len(self.edge_countset):
            raise ValueError(
                "Number of sub-lists in the list of priors "
                "must agree with the number of situations."
            )

        for node_idx, node_priors in enumerate(prior):
            if len(node_priors) != len(self.edge_countset[node_idx]):
                raise ValueError(
                    "The length of each sub-list in the list of priors "
                    "must agree with the number of edges emanating from "
                    "its corresponding situation."
                )
            for p in node_priors:
                if p < 0:
                    raise ValueError("All priors must be non-negative.")

    def _store_params(self, prior, alpha, hyperstage) -> None:
        """User has passed in AHC params, this function processes them,
        and generates any default AHC params if required."""
        if prior:
            if alpha:
                self.alpha = None
                logging.warning("Params Warning!! When prior is given, " +
                                "alpha is not required!")
            self._check_prior(prior)
            self.prior = prior
        else:
            if alpha is None:
                self.alpha = self._calculate_default_alpha()
                logging.warning("Params Warning!! Neither prior nor alpha " +
                                "were provided. Using default alpha " +
                                "value of %d.", self.alpha)
            else:
                self.alpha = alpha

            # No matter what alpha is, generate default prior
            self.prior = self._create_default_prior(self.alpha)

        if hyperstage is None:
            self.hyperstage = self._create_default_hyperstage()
        else:
            self._check_hyperstages(hyperstage)
            self.hyperstage = hyperstage

    def _calculate_default_alpha(self) -> int:
        """If no alpha is given, a default value is calculated.
        The value is calculated by determining the maximum number
        of categories that any one variable has"""
        logger.info("Calculating default prior")
        max_count = max(list(self.categories_per_variable.values()))
        return max_count

    def _create_default_prior(self, alpha) -> list:
        """default prior set for the AHC method using the mass conservation property.
        That is, the alpha param is the phantom sample starting at
        the root, and it is spread equally across all edges along the tree.
        (see chapter 5 of Collazo, Gorgen & Smith 'Chain Event Graphs', 2018)
        The prior is a list of lists. Each list gives the prior along the
        edges of a specific situation.
        Indexed same as self.situations & self.egde_countset"""

        logger.info("Generating default prior")
        default_prior = [0] * len(self.situations)
        sample_size_at_node = dict()

        # Root node is assigned phantom sample (alpha)
        if isinstance(alpha, float):
            alpha = Fraction.from_float(alpha)
        elif isinstance(alpha, int) or isinstance(alpha, str):
            alpha = Fraction.from_float(float(alpha))
        else:
            raise TypeError("Prior generator param alpha is in a strange format.\
                            ..")

        sample_size_at_node[self.root] = alpha

        for node_idx, node in enumerate(self.situations):
            # How many nodes emanate from the current node?
            number_of_emanating_nodes = self.out_degree[node]

            # Divide the sample size from the current node equally among
            # emanating nodes
            equal_distribution_of_sample = \
                sample_size_at_node[node] / number_of_emanating_nodes
            default_prior[node_idx] = \
                [equal_distribution_of_sample] * number_of_emanating_nodes

            relevant_terminating_nodes = [
                edge[1] for edge in list(self.edges) if edge[0] == node
            ]

            for terminating_node in relevant_terminating_nodes:
                sample_size_at_node[terminating_node] = \
                 equal_distribution_of_sample

        return default_prior

    def _create_default_hyperstage(self) -> list:
        '''Generates default hyperstage for the AHC method.
        A hyperstage is a list of lists such that two situaions can be in the
        same stage only if there are elements of the same list for some list
        in the hyperstage.
        The default is to allow all situations with the same number of
        outgoing edges and the same edge labels to be in a common list. '''
        logger.info("Creating default hyperstage")
        hyperstage = []
        info_of_edges = []

        for node in self.situations:
            labels = [
                edge[2] for edge in self.edges
                if edge[0] == node
            ]
            labels.sort()

            info_of_edges.append([self.out_degree[node], labels])

        sorted_info = []
        for info in info_of_edges:
            if info not in sorted_info:
                sorted_info.append(info)

        for info in sorted_info:
            situations_with_value_edges = []
            for idx, situation in enumerate(self.situations):
                if info_of_edges[idx] == info:
                    situations_with_value_edges.append(situation)
            hyperstage = hyperstage + [situations_with_value_edges]

        return hyperstage

    def _create_edge_countset(self) -> list:
        '''Each element of list contains a list with counts along edges emanating from
        a specific situation. Indexed same as self.situations'''
        logger.info("Creating edge countset")
        edge_countset = []

        for node in self.situations:
            edge_countset.append([
                count for edge, count in self.edge_counts.items()
                if edge[0] == node
            ])
        return edge_countset

    def _calculate_lg_of_sum(self, array) -> float:
        '''function to calculate log gamma of the sum of an array'''
        array = [float(x) for x in array]
        return scipy.special.gammaln(sum(array))

    def _calculate_sum_of_lg(self, array) -> float:
        '''function to calculate log gamma of each element of an array'''
        return sum([scipy.special.gammaln(float(x)) for x in array])

    def _calculate_initial_loglikelihood(self, prior, posterior) -> float:
        '''calculating log likelihood given a prior and posterior'''
        # Calculate prior contribution
        logger.info("Calculating initial loglikelihood")

        pri_lg_of_sum = [
            self._calculate_lg_of_sum(elem) for elem in prior
        ]
        pri_sum_of_lg = [
            self._calculate_sum_of_lg(elem) for elem in prior
        ]
        pri_contribution = list(map(sub, pri_lg_of_sum, pri_sum_of_lg))

        # Calculate posterior contribution
        post_lg_of_sum = [
            self._calculate_lg_of_sum(elem) for elem in posterior
        ]
        post_sum_of_lg = [
            self._calculate_sum_of_lg(elem) for elem in posterior
        ]
        post_contribution = list(map(sub, post_sum_of_lg, post_lg_of_sum))

        sum_pri_contribution = sum(pri_contribution)
        sum_post_contribution = sum(post_contribution)
        return (sum_pri_contribution + sum_post_contribution)

    def _calculate_bayes_factor(self, prior1, posterior1,
                                prior2, posterior2) -> float:
        '''calculates the bayes factor comparing two models which differ in
        only one stage'''
        if all(prior1) == 0 or all(prior2) == 0:
            return -np.inf
        else:
            new_prior = list(map(add, prior1, prior2))
            new_posterior = list(map(add, posterior1, posterior2))
            return (
                self._calculate_lg_of_sum(new_prior)
                - self._calculate_lg_of_sum(prior1)
                - self._calculate_lg_of_sum(prior2)
                - self._calculate_lg_of_sum(new_posterior)
                + self._calculate_lg_of_sum(posterior1)
                + self._calculate_lg_of_sum(posterior2)
                + self._calculate_sum_of_lg(new_posterior)
                - self._calculate_sum_of_lg(posterior1)
                - self._calculate_sum_of_lg(posterior2)
                - self._calculate_sum_of_lg(new_prior)
                + self._calculate_sum_of_lg(prior1)
                + self._calculate_sum_of_lg(prior2)
            )
    def _sort_list(self, list_of_tuples) -> list:
        '''function to sort a list of lists to remove repetitions'''

        for l1_idx in range(0, len(list_of_tuples)):
            for l2_idx in range(l1_idx+1, len(list_of_tuples)):
                tup_1 = list_of_tuples[l1_idx]
                tup_2 = list_of_tuples[l2_idx]
                tups_intersect = set(tup_1) & set(tup_2)

                if tups_intersect:
                    union = tuple(set(tup_1) | set(tup_2))
                    list_of_tuples[l1_idx] = []
                    list_of_tuples[l2_idx] = union

        new_list_of_tuples = [
            elem for elem in list_of_tuples
            if elem != []
        ]

        if new_list_of_tuples == list_of_tuples:
            return new_list_of_tuples
        else:
            return self._sort_list(new_list_of_tuples)

    def _calculate_mean_posterior_probs(
        self,
        merged_situations: List,
        posteriors: List,
    ) -> List:
        '''Iterates through array of lists, calculates mean
        posterior probabilities'''
        mean_posterior_probs = []

        for sit in self.situations:
            if sit not in list(chain(*merged_situations)):
                merged_situations.append((sit,))
        for stage in merged_situations:
            for sit in stage:
                sit_idx = self.situations.index(sit)
                if all(posteriors[sit_idx]) != 0:
                    stage_probs = posteriors[sit_idx]
                    break
                else:
                    stage_probs = []

            total = sum(stage_probs)
            mean_posterior_probs.append(
                [round(elem/total, 3) for elem in stage_probs]
            )

        return mean_posterior_probs

    def _independent_hyperstage_generator(
            self, hyperstage: List[List]) -> List[List[List]]:
        """Spit out the next hyperstage that can be dealt with
        independently."""
        new_hyperstages = [[hyperstage[0]]]

        for sublist in hyperstage[1:]:
            hs_to_add = [sublist]

            for hs in new_hyperstages.copy():
                for other_sublist in hs:
                    if not set(other_sublist).isdisjoint(set(sublist)):
                        hs_to_add.extend(hs)
                        new_hyperstages.remove(hs)

            new_hyperstages.append(hs_to_add)

        return new_hyperstages

    def _execute_AHC(self, hyperstage=None) -> Tuple[List, float, List]:
        """finds all subsets and scores them"""
        if hyperstage is None:
            hyperstage = deepcopy(self.hyperstage)

        priors = deepcopy(self.prior_list)
        posteriors = deepcopy(self.posterior_list)

        loglikelihood = self._calculate_initial_loglikelihood(
            priors, posteriors
        )

        merged_situation_list = []

        # Which list in hyperstage have only 1 edge coming out of them
        # For that list, add the list to the merged situation list, and
        # add together their priors/posteriors, and remove it from the
        # hyperstage
        for sub_hyper in deepcopy(hyperstage):
            if self.out_degree[sub_hyper[0]] == 1:
                merged_situation_list.append(tuple(sub_hyper))
                indexes = [self.situations.index(situ) for situ in sub_hyper]
                if len(indexes) == 1:
                    sub_hyper_priors = itemgetter(*indexes)(priors)
                    sub_hyper_posteriors = itemgetter(*indexes)(posteriors)
                else:
                    sub_hyper_priors = list(chain(
                        *itemgetter(*indexes)(priors)
                    ))
                    sub_hyper_posteriors = list(chain(
                        *itemgetter(*indexes)(posteriors)
                    ))
                for loop_idx, p_index in enumerate(indexes):
                    if loop_idx == 0:
                        priors[p_index] = [sum(sub_hyper_priors)]
                        posteriors[p_index] = [sum(sub_hyper_posteriors)]
                    else:
                        priors[p_index] = [0]
                        posteriors[p_index] = [0]

                hyperstage.remove(sub_hyper)

        hyperstage_combinations = [
            item for sub_hyper in hyperstage
            for item in combinations(sub_hyper, 2)
        ]
        while True:
            newscores_list = [self._calculate_bayes_factor(
                priors[self.situations.index(sub_hyper[0])],
                posteriors[self.situations.index(sub_hyper[0])],
                priors[self.situations.index(sub_hyper[1])],
                posteriors[self.situations.index(sub_hyper[1])],
            ) for sub_hyper in hyperstage_combinations]
            #get hyperstage_combinations that have a score > 0
            comb = [hyperstage_combinations[i] for i in range(len(hyperstage_combinations)) if newscores_list[i] > 0]
            print(comb)
            # print(newscores_list)
            local_score = max(newscores_list)

            if local_score > 0:
                local_merged = hyperstage_combinations[
                    newscores_list.index(local_score)
                ]
                merge_situ_1, merge_situ_2 = local_merged
                merge_situ_1_idx = self.situations.index(merge_situ_1)
                merge_situ_2_idx = self.situations.index(merge_situ_2)
                merged_situation_list.append(local_merged)

                priors[merge_situ_1_idx] = list(
                    map(
                        add,
                        priors[merge_situ_1_idx],
                        priors[merge_situ_2_idx]
                    )
                )
                posteriors[merge_situ_1_idx] = list(
                    map(
                        add,
                        posteriors[merge_situ_1_idx],
                        posteriors[merge_situ_2_idx]
                    )
                )
                priors[merge_situ_2_idx] = (
                    [0] * len(priors[merge_situ_1_idx]))
                posteriors[merge_situ_2_idx] = (
                    [0] * len(posteriors[merge_situ_1_idx]))

                loglikelihood += local_score
            else:
                break

        merged_situation_list = self._sort_list(merged_situation_list)
        for sit in self.situations:
            if sit not in list(chain(*merged_situation_list)):
                merged_situation_list.append((sit,))

        self._apply_mean_posterior_probs(
            merged_situations=merged_situation_list,
            mean_posterior_probs=_calculate_mean_posterior_probs(
                self.situations, merged_situation_list, posteriors
            ),
        )

        return loglikelihood, merged_situation_list

    def _execute_aMPC(self, hyperstage=None) -> Tuple[List, float, List]:
        """finds all subsets and scores them"""
        if hyperstage is None:
            hyperstage = deepcopy(self.hyperstage)

        priors = deepcopy(self.prior_list)
        posteriors = deepcopy(self.posterior_list)

        loglikelihood = self._calculate_initial_loglikelihood(
            priors, posteriors
        )

        merged_situation_list = []

        # Which list in hyperstage have only 1 edge coming out of them
        # For that list, add the list to the merged situation list, and
        # add together their priors/posteriors, and remove it from the
        # hyperstage
        for sub_hyper in deepcopy(hyperstage):
            if self.out_degree[sub_hyper[0]] == 1:
                merged_situation_list.append(tuple(sub_hyper))
                indexes = [self.situations.index(situ) for situ in sub_hyper]
                if len(indexes) == 1:
                    sub_hyper_priors = itemgetter(*indexes)(priors)
                    sub_hyper_posteriors = itemgetter(*indexes)(posteriors)
                else:
                    sub_hyper_priors = list(chain(
                        *itemgetter(*indexes)(priors)
                    ))
                    sub_hyper_posteriors = list(chain(
                        *itemgetter(*indexes)(posteriors)
                    ))
                for loop_idx, p_index in enumerate(indexes):
                    if loop_idx == 0:
                        priors[p_index] = [sum(sub_hyper_priors)]
                        posteriors[p_index] = [sum(sub_hyper_posteriors)]
                    else:
                        priors[p_index] = [0]
                        posteriors[p_index] = [0]

                hyperstage.remove(sub_hyper)

        hyperstage_combinations = [
            item for sub_hyper in hyperstage
            for item in combinations(sub_hyper, 2)
        ]
        #instead of calulating scores choose distance and then calulate for that distance
        while True:
            newscores_list = [self._calculate_bayes_factor(
                priors[self.situations.index(sub_hyper[0])],
                posteriors[self.situations.index(sub_hyper[0])],
                priors[self.situations.index(sub_hyper[1])],
                posteriors[self.situations.index(sub_hyper[1])],
            ) for sub_hyper in hyperstage_combinations]

            local_score = max(newscores_list)

            if local_score > 0:
                local_merged = hyperstage_combinations[
                    newscores_list.index(local_score)
                ]
                merge_situ_1, merge_situ_2 = local_merged
                merge_situ_1_idx = self.situations.index(merge_situ_1)
                merge_situ_2_idx = self.situations.index(merge_situ_2)
                merged_situation_list.append(local_merged)

                priors[merge_situ_1_idx] = list(
                    map(
                        add,
                        priors[merge_situ_1_idx],
                        priors[merge_situ_2_idx]
                    )
                )
                posteriors[merge_situ_1_idx] = list(
                    map(
                        add,
                        posteriors[merge_situ_1_idx],
                        posteriors[merge_situ_2_idx]
                    )
                )
                priors[merge_situ_2_idx] = (
                    [0] * len(priors[merge_situ_1_idx]))
                posteriors[merge_situ_2_idx] = (
                    [0] * len(posteriors[merge_situ_1_idx]))

                loglikelihood += local_score
            else:
                break

        merged_situation_list = self._sort_list(merged_situation_list)
        for sit in self.situations:
            if sit not in list(chain(*merged_situation_list)):
                merged_situation_list.append((sit,))

        self._apply_mean_posterior_probs(
            merged_situations=merged_situation_list,
            mean_posterior_probs=_calculate_mean_posterior_probs(
                self.situations, merged_situation_list, posteriors
            ),
        )

        return loglikelihood, merged_situation_list

    def _execute_MPC(self, hyperstage=None) -> Tuple[List, float, List]:
        """finds all subsets and scores them"""
        if hyperstage is None:
            hyperstage = deepcopy(self.hyperstage)

    
        priors = deepcopy(self.prior_list)
        posteriors = deepcopy(self.posterior_list)

        loglikelihood = self._calculate_initial_loglikelihood(
            priors, posteriors
        )
        merged_situation_list = []
        G = nx.Graph()


        # # Which list in hyperstage have only 1 edge coming out of them
        # # For that list, add the list to the merged situation list, and
        # # add together their priors/posteriors, and remove it from the
        # # hyperstage
        for sub_hyper in deepcopy(hyperstage):
            if self.out_degree[sub_hyper[0]] == 1:
                merged_situation_list.append(tuple(sub_hyper))
                indexes = [self.situations.index(situ) for situ in sub_hyper]
                if len(indexes) == 1:
                    sub_hyper_priors = itemgetter(*indexes)(priors)
                    sub_hyper_posteriors = itemgetter(*indexes)(posteriors)
                else:
                    sub_hyper_priors = list(chain(
                        *itemgetter(*indexes)(priors)
                    ))
                    sub_hyper_posteriors = list(chain(
                        *itemgetter(*indexes)(posteriors)
                    ))
                for loop_idx, p_index in enumerate(indexes):
                    if loop_idx == 0:
                        priors[p_index] = [sum(sub_hyper_priors)]
                        posteriors[p_index] = [sum(sub_hyper_posteriors)]
                    else:
                        priors[p_index] = [0]
                        posteriors[p_index] = [0]

                hyperstage.remove(sub_hyper)

        hyperstage_combinations = [
            item for sub_hyper in hyperstage
            for item in combinations(sub_hyper, 2)
        ]
        while True:
            newscores_list = [self._calculate_bayes_factor(
                priors[self.situations.index(sub_hyper[0])],
                posteriors[self.situations.index(sub_hyper[0])],
                priors[self.situations.index(sub_hyper[1])],
                posteriors[self.situations.index(sub_hyper[1])],
             ) if sub_hyper not in merged_situation_list else -np.inf for sub_hyper in hyperstage_combinations ]
            local_score = max(newscores_list)
            if local_score > 0:
                local_merged = hyperstage_combinations[
                    newscores_list.index(local_score)
                ]
                merge_situ_1, merge_situ_2 = local_merged
                merge_situ_1_idx = self.situations.index(merge_situ_1)
                merge_situ_2_idx = self.situations.index(merge_situ_2)
                merged_situation_list.append(local_merged)
                G.add_edge(*local_merged)
                priors[merge_situ_1_idx] = list(
                    map(
                        add,
                        priors[merge_situ_1_idx],
                        priors[merge_situ_2_idx]
                    )
                )
                posteriors[merge_situ_1_idx] = list(
                    map(
                        add,
                        posteriors[merge_situ_1_idx],
                        posteriors[merge_situ_2_idx]
                    )
                )

                joined= list(nx.node_connected_component(G, merge_situ_1))
                for situ in joined:
                    merge_situ_all_idx = self.situations.index(situ)
                    priors[merge_situ_all_idx] = priors[merge_situ_1_idx]
                    posteriors[merge_situ_all_idx] = posteriors[merge_situ_1_idx]

                loglikelihood += local_score
            else:
                break

        merged_situation_list = self._sort_list(merged_situation_list)
        for sit in self.situations:
            if sit not in list(chain(*merged_situation_list)):
                merged_situation_list.append((sit,))

        self._apply_mean_posterior_probs(
            merged_situations=merged_situation_list,
            mean_posterior_probs=_calculate_mean_posterior_probs(
                self.situations, merged_situation_list, posteriors
            ),
        )

        return loglikelihood, merged_situation_list

    def _execute_wHAC(self, hyperstage=None) -> Tuple[List, float, List]:
        """finds all subsets and scores them"""
        if hyperstage is None:
            hyperstage = deepcopy(self.hyperstage)

        priors = deepcopy(self.prior_list)
        posteriors = deepcopy(self.posterior_list)

        loglikelihood = self._calculate_initial_loglikelihood(
            priors, posteriors
        )

        merged_situation_list = []

        # Which list in hyperstage have only 1 edge coming out of them
        # For that list, add the list to the merged situation list, and
        # add together their priors/posteriors, and remove it from the
        # hyperstage
        for sub_hyper in deepcopy(hyperstage):
            if self.out_degree[sub_hyper[0]] == 1:
                merged_situation_list.append(tuple(sub_hyper))
                indexes = [self.situations.index(situ) for situ in sub_hyper]
                if len(indexes) == 1:
                    sub_hyper_priors = itemgetter(*indexes)(priors)
                    sub_hyper_posteriors = itemgetter(*indexes)(posteriors)
                else:
                    sub_hyper_priors = list(chain(
                        *itemgetter(*indexes)(priors)
                    ))
                    sub_hyper_posteriors = list(chain(
                        *itemgetter(*indexes)(posteriors)
                    ))
                for loop_idx, p_index in enumerate(indexes):
                    if loop_idx == 0:
                        priors[p_index] = [sum(sub_hyper_priors)]
                        posteriors[p_index] = [sum(sub_hyper_posteriors)]
                    else:
                        priors[p_index] = [0]
                        posteriors[p_index] = [0]

                hyperstage.remove(sub_hyper)

        hyperstage_combinations = [
            item for sub_hyper in hyperstage
            for item in combinations(sub_hyper, 2)
        ]

        while True:
            newscores_list = [self._calculate_bayes_factor(
                priors[self.situations.index(sub_hyper[0])],
                posteriors[self.situations.index(sub_hyper[0])],
                priors[self.situations.index(sub_hyper[1])],
                posteriors[self.situations.index(sub_hyper[1])],
            ) for sub_hyper in hyperstage_combinations]
            if newscores_list == []:
                break
            if np.isnan(newscores_list).all():
                break
            exp_newscores = [np.exp(x) for x in newscores_list]
            #break if all values are zero
            if exp_newscores == [0] * len(exp_newscores):
                break
            max_local_score = max(exp_newscores)
            if (max_local_score > 1 or random.random() < 0.05):# and max_local_score != -np.inf:
                # exp_newscores = [np.exp(x) for x in newscores_list]
                # exp_newscores = [0 if x != x else x for x in exp_newscores]
                local_score = random.choices(
                    population=newscores_list,
                    weights=exp_newscores,
                    k=1)
                if local_score == [-1.0]:
                    print(exp_newscores)
                # else:
                local_merged = hyperstage_combinations[
                    newscores_list.index(*local_score)
                ]
                merge_situ_1, merge_situ_2 = local_merged
                merge_situ_1_idx = self.situations.index(merge_situ_1)
                merge_situ_2_idx = self.situations.index(merge_situ_2)
                merged_situation_list.append(local_merged)

                priors[merge_situ_1_idx] = list(
                    map(
                        add,
                        priors[merge_situ_1_idx],
                        priors[merge_situ_2_idx]
                    )
                )
                posteriors[merge_situ_1_idx] = list(
                    map(
                        add,
                        posteriors[merge_situ_1_idx],
                        posteriors[merge_situ_2_idx]
                    )
                )
                priors[merge_situ_2_idx] = (
                    [0] * len(priors[merge_situ_1_idx]))
                posteriors[merge_situ_2_idx] = (
                    [0] * len(posteriors[merge_situ_1_idx]))

                loglikelihood += local_score
            else:
                break

        merged_situation_list = self._sort_list(merged_situation_list)
        for sit in self.situations:
            if sit not in list(chain(*merged_situation_list)):
                merged_situation_list.append((sit,))

        self._apply_mean_posterior_probs(
            merged_situations=merged_situation_list,
            mean_posterior_probs=_calculate_mean_posterior_probs(
                self.situations, merged_situation_list, posteriors
            ),
        )

        return loglikelihood, merged_situation_list

    def _execute_full(self, hyperstage=None) -> Tuple[List, float, List]:
        """finds all subsets and scores them"""
        if hyperstage is None:
            hyperstage = deepcopy(self.hyperstage)

        priors = deepcopy(self.prior_list)
        posteriors = deepcopy(self.posterior_list)

        loglikelihood = self._calculate_initial_loglikelihood(
            priors, posteriors
        )

        merged_situation_list = []

        # Which list in hyperstage have only 1 edge coming out of them
        # For that list, add the list to the merged situation list, and
        # add together their priors/posteriors, and remove it from the
        # hyperstage
        for sub_hyper in deepcopy(hyperstage):
            if self.out_degree[sub_hyper[0]] == 1:
                merged_situation_list.append(tuple(sub_hyper))
                indexes = [self.situations.index(situ) for situ in sub_hyper]
                if len(indexes) == 1:
                    sub_hyper_priors = itemgetter(*indexes)(priors)
                    sub_hyper_posteriors = itemgetter(*indexes)(posteriors)
                else:
                    sub_hyper_priors = list(chain(
                        *itemgetter(*indexes)(priors)
                    ))
                    sub_hyper_posteriors = list(chain(
                        *itemgetter(*indexes)(posteriors)
                    ))
                for loop_idx, p_index in enumerate(indexes):
                    if loop_idx == 0:
                        priors[p_index] = [sum(sub_hyper_priors)]
                        posteriors[p_index] = [sum(sub_hyper_posteriors)]
                    else:
                        priors[p_index] = [0]
                        posteriors[p_index] = [0]

                hyperstage.remove(sub_hyper)

        hyperstage_combinations = [
            item for sub_hyper in hyperstage
            for item in combinations(sub_hyper, 2)
        ]

        while True:
            newscores_list = [self._calculate_bayes_factor(
                priors[self.situations.index(sub_hyper[0])],
                posteriors[self.situations.index(sub_hyper[0])],
                priors[self.situations.index(sub_hyper[1])],
                posteriors[self.situations.index(sub_hyper[1])],
            ) if sub_hyper not in merged_situation_list
            else -np.inf
            for sub_hyper in hyperstage_combinations ]
            local_score = max(newscores_list)
            if local_score == -np.inf:
                 break
            # if np.isnan(newscores_list).all():
            #     break
            # max_local_score = max(newscores_list)
            # # if max_local_score > 0:
            # exp_newscores = [np.exp(x) for x in newscores_list]
            # exp_newscores = [0 if x != x else x for x in exp_newscores]
            # local_score = random.choices(
            #     population=newscores_list,
            #     weights=exp_newscores,
            #     k=1)
            local_merged = hyperstage_combinations[
                newscores_list.index(local_score)
            ]
            merge_situ_1, merge_situ_2 = local_merged
            merge_situ_1_idx = self.situations.index(merge_situ_1)
            merge_situ_2_idx = self.situations.index(merge_situ_2)
            merged_situation_list.append(local_merged)

            priors[merge_situ_1_idx] = list(
                map(
                    add,
                    priors[merge_situ_1_idx],
                    priors[merge_situ_2_idx]
                )
            )
            posteriors[merge_situ_1_idx] = list(
                map(
                    add,
                    posteriors[merge_situ_1_idx],
                    posteriors[merge_situ_2_idx]
                )
            )

            # priors[merge_situ_2_idx] = priors[merge_situ_1_idx]
            # posteriors[merge_situ_2_idx] = posteriors[merge_situ_1_idx]
            priors[merge_situ_2_idx] = (
                [0] * len(priors[merge_situ_1_idx]))
            posteriors[merge_situ_2_idx] = (
                [0] * len(posteriors[merge_situ_1_idx]))

            loglikelihood += local_score


        merged_situation_list = self._sort_list(merged_situation_list)
        for sit in self.situations:
            if sit not in list(chain(*merged_situation_list)):
                merged_situation_list.append((sit,))

        self._apply_mean_posterior_probs(
            merged_situations=merged_situation_list,
            mean_posterior_probs=_calculate_mean_posterior_probs(
                self.situations, merged_situation_list, posteriors
            ),
        )

        return loglikelihood, merged_situation_list


    def _mark_nodes_with_stage_number(self, merged_situations):
        """AHC algorithm creates a list of indexes to the situations list.
        This function takes those indexes and creates a new list which is
        in a string representation of nodes."""
        self._sort_count = 0
        stage_count = 0
        for stage in merged_situations:
            if len(stage) > 1:
                for node in stage:
                    self.nodes[node]['stage'] = stage_count
                stage_count += 1

    def _generate_colours_for_situations(self, merged_situations, colour_list):
        """Colours each stage of the tree with an individual colour"""
        num_colours = len([m for m in merged_situations if len(m) > 1])
        if colour_list is None:
            stage_colours = Util.generate_colours(num_colours)
        else:
            stage_colours = colour_list
            if len(colour_list) < num_colours:
                raise IndexError(
                    f"The number of colours provided ({len(colour_list)}) is "
                    "less than the number of distinct colours required "
                    f"({num_colours})."
                )
        self._stage_colours = stage_colours
        for node in self.nodes:
            try:
                stage = self.nodes[node]['stage']
                self.nodes[node]['colour'] = stage_colours[stage]
            except KeyError:
                self.nodes[node]['colour'] = 'lightgrey'

    def calculate_AHC_transitions(self, prior=None,
                                  alpha=None, hyperstage=None,
                                  colour_list=None):
        '''Bayesian Agglommerative Hierarchical Clustering algorithm
        implementation. It returns a list of lists of the situations which
        have been merged together, the likelihood of the final model and
        the mean posterior conditional probabilities of the stages.

        User can specify a list of colours to be used for stages. Otherwise,
        colours evenly spaced around the colour spectrum are used.'''
        logger.info("\n\n --- Starting AHC Algorithm ---")

        self._store_params(prior, alpha, hyperstage)

        loglikelihood, merged_situations = (
            self._execute_AHC())

        self._mark_nodes_with_stage_number(merged_situations)

        self._generate_colours_for_situations(merged_situations, colour_list)

        self.ahc_output = {
            "Merged Situations": merged_situations,
            "Loglikelihood": loglikelihood,
        }
        return self.ahc_output

    def calculate_wHAC_transitions(self, prior=None,
                                  alpha=None, hyperstage=None,
                                  colour_list=None):
        '''Bayesian Agglommerative Hierarchical Clustering algorithm
        implementation. It returns a list of lists of the situations which
        have been merged together, the likelihood of the final model and
        the mean posterior conditional probabilities of the stages.

        User can specify a list of colours to be used for stages. Otherwise,
        colours evenly spaced around the colour spectrum are used.'''
        logger.info("\n\n --- Starting AHC Algorithm ---")

        self._store_params(prior, alpha, hyperstage)

        loglikelihood, merged_situations = (
            self._execute_wHAC())

        self._mark_nodes_with_stage_number(merged_situations)

        self._generate_colours_for_situations(merged_situations, colour_list)

        self.ahc_output = {
            "Merged Situations": merged_situations,
            "Loglikelihood": loglikelihood,
        }
        return self.ahc_output
    def calculate_aMPC_transitions(self, prior=None,
                                  alpha=None, hyperstage=None,
                                  colour_list=None):
        '''Bayesian Agglommerative Hierarchical Clustering algorithm
        implementation. It returns a list of lists of the situations which
        have been merged together, the likelihood of the final model and
        the mean posterior conditional probabilities of the stages.

        User can specify a list of colours to be used for stages. Otherwise,
        colours evenly spaced around the colour spectrum are used.'''
        logger.info("\n\n --- Starting AHC Algorithm ---")

        self._store_params(prior, alpha, hyperstage)

        loglikelihood, merged_situations = (
            self._execute_aMPC())

        self._mark_nodes_with_stage_number(merged_situations)

        self._generate_colours_for_situations(merged_situations, colour_list)

        self.ahc_output = {
            "Merged Situations": merged_situations,
            "Loglikelihood": loglikelihood,
        }
        return self.ahc_output

    def calculate_MPC_transitions(self, prior=None,
                                  alpha=None, hyperstage=None,
                                  colour_list=None):
        '''Bayesian Agglommerative Hierarchical Clustering algorithm
        implementation. It returns a list of lists of the situations which
        have been merged together, the likelihood of the final model and
        the mean posterior conditional probabilities of the stages.

        User can specify a list of colours to be used for stages. Otherwise,
        colours evenly spaced around the colour spectrum are used.'''
        logger.info("\n\n --- Starting AHC Algorithm ---")

        self._store_params(prior, alpha, hyperstage)

        loglikelihood, merged_situations = (
            self._execute_MPC())

        self._mark_nodes_with_stage_number(merged_situations)

        self._generate_colours_for_situations(merged_situations, colour_list)

        self.ahc_output = {
            "Merged Situations": merged_situations,
            "Loglikelihood": loglikelihood,
        }
        return self.ahc_output

    def calculate_full_transitions(self, prior=None,
                                  alpha=None, hyperstage=None,
                                  colour_list=None):
        '''Bayesian Agglommerative Hierarchical Clustering algorithm
        implementation. It returns a list of lists of the situations which
        have been merged together, the likelihood of the final model and
        the mean posterior conditional probabilities of the stages.

        User can specify a list of colours to be used for stages. Otherwise,
        colours evenly spaced around the colour spectrum are used.'''
        logger.info("\n\n --- Starting AHC Algorithm ---")

        self._store_params(prior, alpha, hyperstage)

        loglikelihood, merged_situations = (
            self._execute_full())

        self._mark_nodes_with_stage_number(merged_situations)

        self._generate_colours_for_situations(merged_situations, colour_list)

        self.ahc_output = {
            "Merged Situations": merged_situations,
            "Loglikelihood": loglikelihood,
        }
        return self.ahc_output

    def dot_staged_graph(self, edge_info: str ="count"):
        return self._generate_dot_graph(edge_info=edge_info)

    def create_figure(
        self,
        filename: Optional[str] = None,
        staged: bool = True,
        edge_info: str ="count"
    ) -> Union[Image, None]:
        """Draws the coloured staged tree for the process described by
        the dataset, and saves it to "<filename>.filetype". Supports
        any filetype that graphviz supports. e.g: "event_tree.png" or
        "event_tree.svg" etc.
        """
        if staged:
            try:
                _ = self._ahc_output
                graph = self.dot_staged_graph(edge_info)
                graph_image = super()._create_figure(graph, filename)

            except AttributeError:
                logger.error(
                    "----- PLEASE RUN AHC ALGORITHM before trying to" +
                    " export a staged tree graph -----"
                )
                graph_image = None
        else:
            graph_image = super().create_figure(filename)
        return graph_image

    def _apply_mean_posterior_probs(
        self, merged_situations: List, mean_posterior_probs: List
    ) -> None:
        """Apply the mean posterior probabilities to each edge."""
        for stage_idx, stage in enumerate(merged_situations):
            for sit in stage:
                dst_nodes = list(chain(self.succ[sit]))
                edge_labels = list(chain(*self.succ[sit].values()))
                for edge_idx, label in enumerate(edge_labels):
                    self.edges[
                        (sit, dst_nodes[edge_idx], label)
                    ]["probability"] = mean_posterior_probs[stage_idx][edge_idx]


def _calculate_mean_posterior_probs(
    all_situations: List, merged_situations: List, posteriors: List
) -> List:
    """Given a staged tree, calculate the mean posterior probs."""
    # Add all situations that are not in a stage.
    mean_posterior_probs = []
    for stage in merged_situations:
        for sit in stage:
            sit_idx = all_situations.index(sit)
            if all(posteriors[sit_idx]) != 0:
                stage_posteriors = posteriors[sit_idx]
                break

        total = sum(stage_posteriors)
        mean_posterior_probs.append(
            [
                round(posterior/total, 3)
                for posterior in stage_posteriors
            ]
        )

    return mean_posterior_probs
