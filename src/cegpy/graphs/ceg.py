import pydotplus as pdp
from copy import deepcopy
from ..utilities.util import Util
from IPython.display import Image
from IPython import get_ipython


class ChainEventGraph(object):
    """
    Class: Chain Event Graph

    Input: Staged tree object (StagedTree)
    Output: Chain event graphs
    """
    def __init__(self, staged_tree=None, root=None, sink='w_inf') -> None:
        self.root = root
        self.sink = sink
        self.st = staged_tree
        self.ahc_output = self.st.get_AHC_output().copy()
        evidence_dict = dict(
            edges=dict(
                evidence=dict(),
                paths=set()
            ),
            vertices=dict(
                evidence=set(),
                paths=set()
            )
        )
        self.evidence = dict(
            certain=deepcopy(evidence_dict), uncertain=deepcopy(evidence_dict)
        )
        if self.ahc_output == {}:
            raise ValueError("Run staged tree AHC transitions first.")

        if self.root is None:
            raise(ValueError('Please input the root node!'))
            # self._identify_root_node()

        self.graph = self._create_graph_representation()
        # self._ceg_positions_edges_optimal()

    def get_evidence_dict(self) -> dict:
        return self.evidence

    def add_evidence(self, type_of_evidence, evidence, certain=True):
        """
        Type of evidence can be:
        'edges' or 'vertices'.
        see documentation.
        """
        if certain:
            dict_to_change = self.evidence['certain']
        else:
            dict_to_change = self.evidence['uncertain']

        # if type_of_evidence == 'variables' and\
        #         certain is False:
        #     if len(evidence) > 1:
        #         raise(
        #             ValueError(
        #                 'You can only provide one' +
        #                 'uncertain variable at a time.\n' +
        #                 'See documentation.'
        #             )
        #         )

        if type_of_evidence == 'edges':
            for key, val in evidence.items():
                dict_to_change[type_of_evidence]['evidence'][key] = val
        elif type_of_evidence == 'vertices':
            new_vertex_set = dict_to_change[type_of_evidence]['evidence'].\
                union(evidence)
            dict_to_change[type_of_evidence]['evidence'] = new_vertex_set
        else:
            raise(ValueError("Unknown evidence type.\n \
                should be 'edges' or 'vertices'.\
                see documentation."))

    def _check_evidence_consistency(self, type_of_evidence,
                                    evidence, certain):
        pass

    def get_evidence_str(self) -> str:
        def add_elems_of_dict_to_str(string, dict):
            for key, val in dict.items():
                string += (' %s:\n' % key)
                string += ('   %s\n' % str(val['evidence']))
            return string

        dict_str = 'The evidence you have given is as follows:\n\n'
        dict_str += ' Evidence you are certain of:\n'
        dict_str = add_elems_of_dict_to_str(
            dict_str, self.evidence['certain']
        )

        dict_str += '\n Evidence you are uncertain of:\n'
        dict_str = add_elems_of_dict_to_str(
            dict_str, self.evidence['uncertain']
        )

        return dict_str

    def _create_graph_representation(self) -> dict:
        """
        This function will build a dictionary that represents
        a graph. Constructed from "event tree" representation
        from ceg.trees.event.
        The output will have the form:
        {
            'nodes': {
                'node_1': {
                    'nodes_to_merge': ['n1', 'n2', ...],
                    'colour': '<hex_colour_string>'
                },
                ...
                'node_n': {
                    ...
                }
            },
            'edges': {
                ('src', 'dest'): [
                    {
                        'src': 'n1',
                        'dest': 'n2',
                        'label': 'Edge_Label',
                        'value': 54.0
                    },
                    {
                        <any other edges between
                        the same nodes>
                    },...
                ],
                ...
            }
        }
        """
        graph = {
            "nodes": {},
            "edges": {}
        }
        event_tree = self.st.get_event_tree()
        prior = self._flatten_list_of_lists(self.st.get_prior())
        posterior_probs = self._flatten_list_of_lists(
            self.ahc_output['Mean Posterior Probabilities']
        )
        # Add all nodes, and their children, to the dictionary
        for idx, edge_key in enumerate(event_tree.keys()):
            # edge has form:
            # (('path', 'to', 'label'), ('<node_name>', '<child_name>'))
            edge = self._create_new_edge(
                src=edge_key[1][0],
                dest=edge_key[1][1],
                label=edge_key[0][-1],
                probability=float(posterior_probs[idx]),
                value=(event_tree[edge_key] + float(prior[idx])))

            new_edge_key = edge_key[1]

            # Add src node to graph dict:
            try:
                graph['nodes'][edge['src']]['outgoing_edges'].append(
                    new_edge_key
                )
            except KeyError:
                root = True if edge['src'] == self.root else False

                graph['nodes'][edge['src']] = \
                    self._create_new_node(
                        colour=self.ahc_output['Node Colours'][edge['src']],
                        outgoing_edges=[new_edge_key],
                        root=root
                    )

            # Add dest node
            try:
                graph['nodes'][edge['dest']]['incoming_edges'].append(
                    new_edge_key
                )
            except KeyError:
                graph['nodes'][edge['dest']] = \
                    self._create_new_node(
                        colour=self.ahc_output['Node Colours'][edge['dest']],
                        incoming_edges=[new_edge_key]
                    )

            # Add edge to graph dict:
            try:
                graph['edges'][new_edge_key].append(edge)
            except KeyError:
                graph['edges'][new_edge_key] = []
                graph['edges'][new_edge_key].append(edge)

        self._update_distances_of_nodes_to_sink(
            graph, self.st.get_leaves().copy()
        )
        return graph

    def _update_distances_of_nodes_to_sink(self, graph, sinks) -> None:
        """
        Provided a graph, and a list of the sink nodes (or leaves),
        this function will update all the nodes in the graph with their
        maximum distance to sink.
        """
        def calc_dist_of_node_to_sink(
                node_dict, edge_dict, node) -> int:

            distances = []
            for out_edge in node_dict[node]['outgoing_edges']:
                dest = edge_dict[out_edge][0]['dest']
                distances.append(node_dict[dest]['max_dist_to_sink'])
            return max(distances) + 1

        def update_dist_of_source_nodes(
                node_dict, edge_dict, destinations) -> list:

            new_destinations = []
            for node_id in destinations:
                for incoming_edge in nodes[node_id]['incoming_edges']:
                    src_node = edges[incoming_edge][0]['src']
                    dist = calc_dist_of_node_to_sink(
                        node_dict, edge_dict, src_node
                    )
                    nodes[src_node]['max_dist_to_sink'] = dist
                    new_destinations.append(src_node)
            return list(set(new_destinations))

        nodes = graph['nodes']
        edges = graph['edges']
        destinations = sinks

        while destinations != []:
            destinations = update_dist_of_source_nodes(
                nodes, edges, destinations
            )

    def _gen_nodes_with_increasing_distance(self, graph, start=0) -> list:
        distance_dict = {}
        nodes = graph['nodes']
        graph_max_dist = nodes[self.root]['max_dist_to_sink']

        # create empty lists for each distance
        for dist in range(graph_max_dist):
            distance_dict[dist] = []

        # look at all nodes in the graph, and add each node
        # to the list with other nodes of the same max distance
        for node_id, node_data in nodes.items():
            if node_data['root'] is False:
                node_max_dist = node_data['max_dist_to_sink']
                distance_dict[node_max_dist].append(node_id)

        for dist in range(graph_max_dist):
            if dist >= start:
                yield distance_dict[dist]

    def _identify_root_node(self, graph) -> str:
        number_of_roots = 0
        root = ''
        for node in graph['nodes']:
            node_properties = graph['nodes'][node]
            if node_properties['incoming_edges'] == []:
                root = node
                number_of_roots += 1

        if number_of_roots > 1:
            raise(ValueError('Your graph has too many roots!'))
        elif number_of_roots == 1:
            return root
        else:
            raise(ValueError('No graph root was found!'))

    def _flatten_list_of_lists(self, list_of_lists) -> list:
        flat_list = []
        for sublist in list_of_lists:
            flat_list = flat_list + sublist
        return flat_list

    def _create_new_node(self, root=False, sink=False,
                         incoming_edges=[], outgoing_edges=[],
                         dist_to_sink=0, colour='lightgrey') -> dict:
        """
        Generates default format of Node dictionary
        """
        return {
            'root': root,
            'sink': sink,
            'incoming_edges': incoming_edges.copy(),
            'outgoing_edges': outgoing_edges.copy(),
            'max_dist_to_sink': dist_to_sink,
            'colour': colour
        }

    def _create_new_edge(self, src='', dest='', label='',
                         probability=0.0, value=0.0) -> list:
        """
        Generates default format of edge dictionary.
        """
        edge = {
            'src': src,
            'dest': dest,
            'label': label,
            'value': value,
            'probability': probability
        }
        return edge

    def _trim_leaves_from_graph(self, graph) -> list:
        leaves = self.st.get_leaves()
        cut_vertices = []
        graph['nodes'][self.sink] = \
            self._create_new_node(sink=True, dist_to_sink=0)

        # Check which nodes have been identified as leaves
        edges_to_delete = []
        edges_to_add = {}
        for leaf in leaves:

            # In edge list, look for edges that terminate on this leaf
            for edge_list_key in graph['edges'].keys():
                if edge_list_key[1] == leaf:
                    new_edge_list = []
                    new_edge_list_key = (edge_list_key[0], self.sink)
                    # Each edge key may have multiple edges associate with it
                    for edge in graph['edges'][edge_list_key]:
                        new_edge = edge
                        new_edge['dest'] = self.sink
                        new_edge_list.append(new_edge)

                    # remove out of date edges from the dictionary
                    edges_to_delete.append(edge_list_key)

                    cut_vertices.append(new_edge_list_key[0])
                    # clean node dict
                    graph['nodes'][edge_list_key[0]]['outgoing_edges'].remove(
                        edge_list_key
                    )
                    try:
                        # add modified edge to the dictionary
                        edges_to_add[new_edge_list_key] = \
                            edges_to_add[new_edge_list_key] + new_edge_list
                    except KeyError:
                        edges_to_add[new_edge_list_key] = \
                            new_edge_list
                    # add edge to node dict
                    outgoing_edges = \
                        graph['nodes'][new_edge_list_key[0]]['outgoing_edges']
                    outgoing_edges.append(new_edge_list_key)

                    incoming_edges = \
                        graph['nodes'][new_edge_list_key[1]]['incoming_edges']
                    incoming_edges.append(new_edge_list_key)

                    outgoing_edges = list(set(outgoing_edges))
                    graph['nodes'][new_edge_list_key[0]]['outgoing_edges'] = \
                        outgoing_edges

                    incoming_edges = list(set(incoming_edges))
                    graph['nodes'][new_edge_list_key[1]]['incoming_edges'] = \
                        incoming_edges

            # remove leaf node from the graph
            del graph['nodes'][leaf]
        # clean up old edges
        for edge in edges_to_delete:
            del graph['edges'][edge]

        graph['edges'] = {**graph['edges'], **edges_to_add}

        return list(set(cut_vertices))

    def _merge_nodes(self) -> dict:
        return self.graph

    def generate_CEG(self):
        '''
        This function takes the output of the AHC algorithm and identifies
        the positions i.e. the vertices of the CEG and the edges of the CEG
        along with their edge labels and edge counts. Here we use the
        algorithm in our paper with the optimal stopping time.
        '''
        def check_vertices_can_be_merged(v1, v2) -> bool:
            vertices_are_equivalent = False
            logic_values = [
                (v1 in stage, v2 in stage)
                for stage in self.ahc_output["Merged Situations"]
            ]
            vertices_are_equivalent = any(
                value == (True, True) for value in logic_values
            )
            return vertices_are_equivalent

        def clean_dict_list(edges, keys_to_keep=[]) -> list:
            """
            For a list of dictionaries, copy the dicts, and remove
            any keys from the dict that are specified."""
            # first copy each of the dictionaries in the list
            new_edges = []
            for edge in edges:
                new_edge = deepcopy(edge)

                keys_to_delete = [
                    key for key in new_edge.keys()
                    if key not in keys_to_keep
                ]

                for key in keys_to_delete:
                    del new_edge[key]

                new_edges.append(new_edge)

            return new_edges

        def get_outgoing_edges(node_dict, edge_dict, node) -> list:
            # obtain a list of all the outgoing edges from this node
            outgoing_edge_keys = node_dict[node]['outgoing_edges']
            outgoing_edges = []
            # Build a list of all the dict objects of the edges leaving
            # this node
            for key in outgoing_edge_keys:
                # remove superfuous data from the dictionary
                try:
                    edges = clean_dict_list(
                        edges=edge_dict[key][:],
                        keys_to_keep=['dest', 'label']
                    )
                except KeyError:
                    pass
                outgoing_edges = outgoing_edges + edges

            # sort the list by the edge label in the dictionary
            outgoing_edges.sort(key=lambda dict: dict['label'])

            return outgoing_edges

        def merge_outgoing_edges(target_edges, source_edges,
                                 edge_dict) -> list:
            """
            Nodes in the same position set can have their outgoing edges
            merged. This translates to their values being added together.
            Target edges are the outgoing edges from the node we are keeping.
            Source edges are the outgoing edges from the node we are merging.
            """
            for src_edge in source_edges:
                src_edge_list = edge_dict[src_edge]
                for src_edg_dtls in src_edge_list:
                    for trg_edge in target_edges:
                        trg_edge_list = edge_dict[trg_edge]
                        for trg_edg_dtls in trg_edge_list:
                            src_lbl = src_edg_dtls['label']
                            trg_lbl = trg_edg_dtls['label']
                            # Edges can be merged!
                            if src_lbl == trg_lbl:
                                trg_edg_dtls['value'] += src_edg_dtls['value']

                del edge_dict[src_edge]
            return target_edges

        def merge_position_set_nodes(node_dict, edge_dict, position_set):
            node_to_keep = position_set[0]
            other_nodes = set(position_set[:])
            other_nodes.remove(node_to_keep)
            keep_outgoing = node_dict[node_to_keep]['outgoing_edges']

            for node in other_nodes:
                # redirect incoming nodes to the node_to_keep
                incoming = node_dict[node]['incoming_edges']
                for edge in incoming:
                    edge_list = edge_dict[edge]
                    for edge_details in edge_list:
                        src_node = edge_details['src']
                        redirected_edge_key = (
                            src_node,
                            node_to_keep
                        )
                        redirected_edge_details = \
                            deepcopy(edge_details)

                        redirected_edge_details['dest'] = \
                            node_to_keep

                        try:
                            edge_dict[redirected_edge_key].append(
                                redirected_edge_details
                            )
                        except KeyError:
                            edge_dict[redirected_edge_key] = [
                                redirected_edge_details
                            ]
                        # Ensure that the src_node of the edge now points
                        # to the node we are keeping
                        node_dict[src_node]['outgoing_edges'].remove(
                            edge
                        )
                        node_dict[src_node]['outgoing_edges'].append(
                            redirected_edge_key
                        )
                    # remove the old edge as it is going to point to a node
                    # that no longer exists
                    del edge_dict[edge]

                # merge outgoing edges for all nodes in same position_set
                other_outgoing = node_dict[node]['outgoing_edges']
                keep_outgoing = merge_outgoing_edges(
                    target_edges=keep_outgoing,
                    source_edges=other_outgoing,
                    edge_dict=edge_dict
                )
                del node_dict[node]

        self._trim_leaves_from_graph(self.graph)
        src_node_gen = self._gen_nodes_with_increasing_distance(
            graph=self.graph,
            start=1
        )
        next_set_of_nodes = next(src_node_gen)

        # Keep track of changes made. When none have been made, set to 0
        changes_made = -1

        nodes = self.graph['nodes']
        edges = self.graph['edges']

        while changes_made != 0:
            position_sets = []
            # Initialise changes made to 0 so that it can be used to terminate
            # the while loop
            changes_made = 0
            while next_set_of_nodes != []:

                for node_1 in next_set_of_nodes:
                    position_set = [node_1]
                    n1_outgoing_edges = get_outgoing_edges(
                        node_dict=nodes,
                        edge_dict=edges,
                        node=node_1
                    )
                    # make a copy of our list, and remove node_1
                    other_src_nodes_of_cut_edges = next_set_of_nodes[:]
                    other_src_nodes_of_cut_edges.remove(node_1)

                    for node_2 in other_src_nodes_of_cut_edges:
                        # check if node_1 and node_2 are ever in the same stage
                        nodes_are_equivalent = check_vertices_can_be_merged(
                            v1=node_1,
                            v2=node_2
                        )
                        if nodes_are_equivalent:
                            n2_outgoing_edges = get_outgoing_edges(
                                node_dict=nodes,
                                edge_dict=edges,
                                node=node_2
                            )

                            if n1_outgoing_edges == n2_outgoing_edges:
                                position_set.append(node_2)

                    if len(position_set) > 1:
                        changes_made += 1

                    position_sets.append(position_set)

                    for node in position_set:
                        next_set_of_nodes.remove(node)

            # Now we have all the stages that need to merge together
            for position_set in position_sets:
                # Only make changes if there are more than one element
                if len(position_set) > 1:
                    merge_position_set_nodes(
                        node_dict=nodes,
                        edge_dict=edges,
                        position_set=position_set
                    )

            try:
                next_set_of_nodes = next(src_node_gen)
            except StopIteration:
                next_set_of_nodes = []

    def create_figure(self, filename):
        filename, filetype = Util.generate_filename_and_mkdir(filename)

        graph = pdp.Dot(graph_type='digraph', rankdir='LR')
        for _, edges in self.graph['edges'].items():
            for edge in edges:
                edge_label = edge['label'] + '\n' + str(edge['probability'])
                graph.add_edge(
                    pdp.Edge(
                        src=edge['src'],
                        dst=edge['dest'],
                        label=edge_label,
                        labelfontcolor='#009933',
                        fontsize='10.0',
                        color='black'
                    )
                )

        for key, node in self.graph['nodes'].items():
            fill_colour = node['colour']

            graph.add_node(
                pdp.Node(
                    name=key,
                    label=key,
                    style='filled',
                    fillcolor=fill_colour
                )
            )

        graph.write(str(filename), format=filetype)

        if get_ipython() is None:
            return None
        else:
            return Image(graph.create_png())

    # def _create_pdp_graph_representation(self) -> dict:
    #     graph = pdp.Dot(graph_type='digraph', rankdir='LR')
    #     event_tree = self.st.get_event_tree()
    #     prior = self._flatten_list_of_lists(self.st.get_prior())

    #     for idx, (key, count) in enumerate(event_tree.items()):
    #         # edge_index = self.edges.index(edge)
    #         path = key[0]
    #         edge = key[1]
    #         edge_details = str(path[-1]) + '\n' + \
    #             str(count + float(prior[idx]))

    #         graph.add_edge(
    #             pdp.Edge(
    #                 edge[0],
    #                 edge[1],
    #                 label=edge_details,
    #                 labelfontcolor="#009933",
    #                 fontsize="10.0",
    #                 color="black"
    #             )
    #         )

    #     for node in self.st.get_nodes():
    #         # if colours:
    #         #     fill_colour = colours[node]
    #         # else:
    #         #     fill_colour = 'lightgrey'

    #         graph.add_node(
    #             pdp.Node(
    #                 name=node,
    #                 label=node,
    #                 style="filled",
    #                 fillcolor='lightgrey'))
    #     edge = graph.get_edge(('s0', 's1'))

    #     edge_label = edge[0].get_label()
    #     print(edge_label)
    #     return graph

    # def _get_edge_details(self, graph, edge_name):
    #     edge = graph.get_edge(edge_name)

    #     try:
    #         edge_label = edge[0].get_label()

    #     except KeyError or IndexError:
    #         edge_label = edge.get_label()
