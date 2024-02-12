"""Collection of tools for processing graphs and dam networks
Used for creation of input files for MOO of dam portfolio selection"""
from typing import List, Dict, Tuple, Any, Sequence, Callable
from itertools import combinations
import pathlib
from dataclasses import dataclass, field
import copy
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# Wrapper for the model network graph
@dataclass
class DamNetwork:
    """Class for loading, querying, updating and plotting graph networks"""
    graph: nx.Graph
    coordinates: Dict[str, Tuple[float, float]] = field(default_factory = dict)
    logging: bool = True
        
    @classmethod
    def from_edges(
            cls, edges: List[Any], **kwargs):
        """Constructs graph from a list of edges"""
        return cls(graph = nx.DiGraph(edges), **kwargs)
    
    def __post_init__(self) -> None:
        """Adds coordinates in graph's node attaributes if the coordinates
        have been provided as an attribute"""
        if self.coordinates:
            nx.set_node_attributes(self.graph, self.coordinates, 'pos')
        
    @property
    def root_nodes(self) -> List[Any]:
        """Finds root nodes, i.e. the nodes without ancestors"""
        root_nodes = [node for node, in_degree in self.graph.in_degree if in_degree == 0]
        return root_nodes
    
    def get_nodes(self, min_indegree: int = 1) -> List[Any]:
        """Finds nodes with in-degree larger equal `min_indegree`"""
        return [
            node for node in sorted(self.graph.nodes) if self.graph.in_degree(node) >= min_indegree]

    def find_edges_containing_node(
            self, node_id: Any, 
            data_key: str | None = 'dam_id') -> List[Tuple[Any, Any, Dict]]:
        """ """
        edge_list: List[Tuple[Any, Any, Dict]] = []
        for _, (u, v, edge_data) in enumerate(sorted(self.graph.edges(data=True))):
            if data_key:
                id = edge_data[data_key]
                if id == node_id:
                    edge_list.append((u, v, edge_data))
            else:
                if node_id in (u, v):
                    edge_list.append((u, v, edge_data))
        return edge_list
    
    def unique_ids(self, data_key = 'dam_id') -> List[int]:
        """ """
        ids = set()
        for (u, v, edge_data) in self.graph.edges(data=True):
            ids.add(edge_data[data_key])
        return list(ids)
    
    @property
    def edges(self) -> List[Any]:
        """Find edges"""
        return self.graph.edges()
        
    @property
    def leaf_nodes(self) -> List[Any]:
        """Leaf nodes are nodes without children"""
        leaf_nodes = [
            node for node, out_degree in self.graph.out_degree if out_degree == 0]
        return leaf_nodes
    
    @property
    def duplicate_edges_by_val(self, val_field: str = 'dam_id') -> List[Any]:
        """Find duplicate edges by field value in edge data"""
        seen_edges = set()
        duplicate_edges = []

        for edge in self.graph.edges(data=True):
            edge_data = edge[2]
            if not edge_data:
                continue
            edge_data_hashable = tuple(sorted(edge_data.items()))
            # Check if this edge data has been seen before
            if edge_data_hashable in seen_edges:
                duplicate_edges.append(edge)
            else:
                seen_edges.add(edge_data_hashable)
        return duplicate_edges
    
    def update_coordinates(self, coordinates: Dict[str, Tuple[float, float]]) -> None:
        """Add coordinates of nodes in a graph"""
        self.coordinates.update(coordinates)
        
    def plot(
            self, font_size: int = 6, edge_data_field: str | None = None, 
            figsize: Tuple[float, float] = (10,8), use_coords: bool = True) -> None:
        """ """
        plt.figure(figsize=(figsize[0], figsize[1]))
        if self.coordinates and use_coords:
            pos = nx.get_node_attributes(self.graph, 'pos')
            #pos = nx.set_node_attributes(self.graph, self.coordinates, 'pos')
        else:
            pos = nx.spring_layout(self.graph)
        nx.draw_networkx_nodes(
            self.graph, pos, cmap=plt.get_cmap('jet'), 
            node_size = 100, edgecolors='grey', node_color='white', alpha=0.6)
        nx.draw_networkx_labels(
            self.graph, pos, alpha=0.8, font_size=font_size, 
            horizontalalignment='left',
            verticalalignment='bottom', font_color='blue')
        nx.draw_networkx_edges(
            self.graph, pos, edge_color='k', arrows=True, alpha=0.7)
        if edge_data_field is not None:
            edge_labels = nx.get_edge_attributes(self.graph, edge_data_field)
            nx.draw_networkx_edge_labels(
                self.graph, pos, edge_labels=edge_labels, font_color="red")
        plt.show()


# Network simplifier
@dataclass
class NetworkSimplifier:
    """Class for simplyfying dam networks"""
    network: DamNetwork
    logging: bool = True

    def __post_init__(self) -> None:
        # Always work on a copy
        self.network = copy.deepcopy(self.network)
        
    def reverse_graph(self, inplace: bool = False) -> nx.Graph:
        """ """
        G_rev = self.network.graph.reverse()
        if inplace:
            self.network.graph = G_rev
        return G_rev
        
    def rename_nodes(self, name_maps: Sequence[Dict], inplace: bool = False) -> nx.Graph:
        """ """
        G = copy.deepcopy(self.network.graph)
        for name_map in name_maps:
            G = nx.relabel_nodes(G, name_map, copy=False)
        if inplace:
            self.network.graph = G
        return G
    
    def simplify(
            self, node_removal_predicate: Callable, 
            inplace: bool = False) -> nx.Graph:
        '''
        Loop over the graph until all nodes that match the supplied predicate 
        have been removed and their incident edges fused.
        TODO: Remove .copy() operators that are not needed.
        '''
        g = self.network.graph
        while any(node_removal_predicate(node) for node in g.nodes):
            g0 = g.copy()
            for node in g.nodes:
                if not node_removal_predicate(node):
                    continue
                    
                if g.is_directed():
                    in_edges_containing_node = list(g0.in_edges(node))
                    out_edges_containing_node = list(g0.out_edges(node))

                    for in_src, _ in in_edges_containing_node:
                        for _, out_dst in out_edges_containing_node:
                            g0.add_edge(in_src, out_dst)
                            if self.logging:
                                print(f"adding edge {in_src} : {out_dst}")
                else:
                    edges_containing_node = g.edges(node)
                    dst_to_link = [e[1] for e in edges_containing_node]
                    dst_pairs_to_link = list(combinations(dst_to_link, r = 2))
                    for pair in dst_pairs_to_link:
                        g0.add_edge(pair[0], pair[1])

                g0.remove_node(node)
                if self.logging:
                    print(f"removing node {node}")
                break
            g = g0.copy()
            if inplace:
                self.network.graph = g
        return g


# Functions for combining dam networks
def combine_disjoint_by_roots(
        network: DamNetwork, by_node: Any | None = None, inplace: bool = True) -> DamNetwork:
    """Function for combining disjoints graph and producing a combined graph
    with a single root ndoe"""
    if not inplace:
        network = copy.deepcopy(network)
    if len(network.root_nodes) == 1:
        print("Network contains only one root node. Nothing to be combined")
        return network
    if by_node:
        if not by_node in network.root_nodes:
            print(f"Specified merge node {by_node} not in root nodes. Using first root node isntead")
            by_node = network.root_nodes[0]
    else:
        by_node = network.root_nodes[0]
    remaining_nodes = network.root_nodes[1:]
    root_child_map = {
        root_node: list(network.graph.successors(root_node)) for 
        root_node in remaining_nodes}
    
    for root_node, children in root_child_map.items():
        for child in children:
            old_edge_data=network.graph.edges[root_node,child]
            network.graph.add_edge(by_node, child, **old_edge_data)
    network.graph.remove_nodes_from(remaining_nodes)
    return network

def combine_by_root_nodes(n1: DamNetwork, n2: DamNetwork) -> DamNetwork:
    """Combines two dam networks and produce a dam network with a single root node"""
    # Work on network copies
    n1 = copy.deepcopy(n1)
    n2 = copy.deepcopy(n2)
    # Step 1: Identify root node(s) of the second dam network
    root_nodes_n2 = n2.root_nodes
    # Step 2: Identify root node of the first dam network
    root_nodes_n1 = n1.root_nodes
    if len(root_nodes_n1) > 1:
        raise ValueError("Newtwork n1 has more than one root. Only one root in n1 is allowed")
    else:
        root_node_n1 = root_nodes_n1[0]
    root_child_map_n2 = {
        root_node : list(n2.graph.successors(root_node)) for root_node in root_nodes_n2}
    # Step 3: Remove root node(s) from the second network
    n2.graph.remove_nodes_from(root_nodes_n2)
    # Step 3: Connect children of removed root node(s) to the root node of the first network
    for root_node_n2, children in root_child_map_n2.items():
        for child in children:
            n1.graph.add_edge(root_node_n1, child)

    combined_graph = nx.compose(n1.graph, n2.graph)
    combined_network = n1
    combined_network.graph = combined_graph
    combined_network.update_coordinates(n2.coordinates)
    return combined_network

def combine_multiple_by_root_nodes(networks: List[DamNetwork]) -> DamNetwork:
    """ """
    dam_network = networks[0]
    for network_ix in range(len(networks)-1):
        dam_network = combine_by_root_nodes(dam_network, networks[network_ix+1])
    return dam_network

# Helper functions working with pywr model network data structures

def get_model_edges(model_path: pathlib.Path) -> List[List[str]]:
    """Finds edges in a Pywr model json file
    Args:
        model_path: Pywr model json file
    Returns: a list of lists, each with two strings describing the names of
        the upstream and downstrem node, respectively
    """
    with open(model_path) as file:
        data = json.load(file)
    edges = data.get('edges')
    return edges


def get_model_coordinates(
        coordinate_file_path: pathlib.Path) -> Dict[str, Tuple[float, float]]:
    """Finds coordinates of Pywr model nodes
    Args:
        coordinate_file_path: path of a JSON with coordinates of Pywr model nodes
    Returns:
        A mapping between node names and their coordinates given as lon, lat tuples
    """
    with open(coordinate_file_path) as file:
        data = json.load(file)
    return data


def pywr_ifc_map_from_csv(file_path: pathlib.Path) -> Dict[str, str]:
    """Creates a map between pywr names and ifc names of pywr model nodes
    Args:
        file_path: csv file with pywr and ifc dam names given in columns
    Returns:
        A map between pywr names and ifc names
    """
    df = pd.read_csv(file_path)
    return df.set_index('pywr_name')['ifc_name'].to_dict()


def ifc_name_to_ifc_id_from_csv(file_path: pathlib.Path) -> Dict[str, int]:
    """Creates a map between ifc name and ifc id of pywr model nodes
    Args:
        file_path: csv file with pywr and ifc dam names given in columns
    Returns:
        A map between ifc names and ifc ids
    """
    df = pd.read_csv(file_path)
    return df.set_index('name')['ifc_id'].to_dict()


def dict_to_json(data: Dict, json_file: str | pathlib.Path) -> None:
    """Dictionary data serializer into json file
    Args:
        data: Any dictionary that is serializable into json string
        json_file: string or pathlib.Path representation of json output file
    Returns None
    """
    with open(json_file, 'w') as file:
        json_string = json.dumps(data, indent=4)
        file.write(json_string)
