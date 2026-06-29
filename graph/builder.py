"""
Graph Builder
─────────────
Converts the IMD 0.25° grid over India into a PyTorch Geometric graph.

Each grid cell = one node.
Nodes are connected to their k nearest geographic neighbors.
Edge features encode distance + direction between nodes.

The graph is built once and saved to disk (graph.pt).
"""

import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
from scipy.spatial import KDTree
import logging

log = logging.getLogger(__name__)


def build_india_graph(
    resolution:  float = 0.5,        # degrees (0.5 for laptop, 0.25 for full)
    lat_bounds:  tuple = (6.5, 38.5),
    lon_bounds:  tuple = (66.5, 100.0),
    k_neighbors: int   = 8,
    save_path:   str   = None,
) -> Data:
    """
    Build a graph over India from a regular lat/lon grid.

    Args:
        resolution  : grid spacing in degrees
        lat_bounds  : (south, north) latitude limits
        lon_bounds  : (west, east)   longitude limits
        k_neighbors : each node connects to k nearest nodes
        save_path   : if given, saves graph.pt here

    Returns:
        torch_geometric.data.Data with:
          .node_pos       (N, 2)  — [lat, lon] of each node
          .edge_index     (2, E)  — sender/receiver pairs
          .edge_attr      (E, 4)  — [dist, dlat, dlon, angle]
          .num_nodes      int
    """
    lats = np.arange(lat_bounds[0], lat_bounds[1] + resolution/2, resolution)
    lons = np.arange(lon_bounds[0], lon_bounds[1] + resolution/2, resolution)

    # Create all grid points
    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")
    node_lats = grid_lat.flatten().astype(np.float32)
    node_lons = grid_lon.flatten().astype(np.float32)
    N = len(node_lats)

    log.info(f"Building graph: {len(lats)} lat × {len(lons)} lon = {N} nodes")

    # ── Build edges using KD-tree ─────────────────────────────────
    # Convert to approximate Cartesian for distance computation
    # (good enough at India scale, no need for haversine)
    coords = np.stack([node_lats, node_lons], axis=1)
    tree   = KDTree(coords)

    # Query k+1 neighbors (first result is the node itself)
    distances, indices = tree.query(coords, k=k_neighbors + 1)

    src_list  = []
    dst_list  = []
    attr_list = []

    for i in range(N):
        for j_idx in range(1, k_neighbors + 1):   # skip self (index 0)
            j    = indices[i, j_idx]
            dist = distances[i, j_idx]

            dlat  = node_lats[j] - node_lats[i]
            dlon  = node_lons[j] - node_lons[i]
            angle = np.arctan2(dlat, dlon)

            # Bidirectional edges
            src_list.append(i);  dst_list.append(j)
            attr_list.append([dist, dlat, dlon, angle])

            src_list.append(j);  dst_list.append(i)
            attr_list.append([dist, -dlat, -dlon, angle + np.pi])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(attr_list,            dtype=torch.float32)
    node_pos   = torch.tensor(coords,               dtype=torch.float32)

    # Normalize edge attributes to [-1, 1]
    edge_attr_norm = edge_attr.clone()
    for c in range(edge_attr.shape[1]):
        col = edge_attr[:, c]
        edge_attr_norm[:, c] = (col - col.mean()) / (col.std() + 1e-8)

    graph = Data(
        node_pos       = node_pos,
        edge_index     = edge_index,
        edge_attr      = edge_attr_norm,
        edge_attr_raw  = edge_attr,
        num_nodes      = N,
    )
    graph.node_lats = torch.tensor(node_lats)
    graph.node_lons = torch.tensor(node_lons)
    graph.lats      = torch.tensor(lats, dtype=torch.float32)
    graph.lons      = torch.tensor(lons, dtype=torch.float32)
    graph.n_lat     = len(lats)
    graph.n_lon     = len(lons)
    graph.resolution = resolution

    log.info(
        f"Graph built: {N} nodes, {edge_index.shape[1]} edges "
        f"(~{edge_index.shape[1]//N} per node)"
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, save_path)
        log.info(f"Graph saved to {save_path}")

    return graph


def load_or_build_graph(config: dict) -> Data:
    """
    Load graph from disk if it exists, otherwise build and save.
    """
    save_path = config["paths"]["graph"]

    if Path(save_path).exists():
        log.info(f"Loading cached graph from {save_path}")
        return torch.load(save_path, weights_only=False)

    log.info("Graph not found — building from config...")
    return build_india_graph(
        resolution  = config["graph"]["resolution"],
        lat_bounds  = tuple(config["graph"]["lat_bounds"]),
        lon_bounds  = tuple(config["graph"]["lon_bounds"]),
        k_neighbors = config["graph"]["k_neighbors"],
        save_path   = save_path,
    )


def grid_to_nodes(data_array, graph: Data) -> torch.Tensor:
    """
    Convert a (T, n_lat, n_lon) xr.DataArray / np.array
    to (T, N_nodes) tensor matching graph node ordering.

    Args:
        data_array : numpy array (T, n_lat, n_lon)
        graph      : the India graph

    Returns:
        torch.Tensor (T, N_nodes)
    """
    T = data_array.shape[0]
    flat = data_array.reshape(T, -1).astype(np.float32)
    return torch.tensor(flat)


def nodes_to_grid(node_tensor: torch.Tensor, graph: Data) -> np.ndarray:
    """
    Convert (T, N_nodes) tensor back to (T, n_lat, n_lon) numpy array.
    Useful for visualization.
    """
    T = node_tensor.shape[0]
    return node_tensor.numpy().reshape(T, graph.n_lat, graph.n_lon)