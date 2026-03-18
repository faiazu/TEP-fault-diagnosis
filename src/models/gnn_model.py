import numpy as np
import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool


# Process graph edges between sensors.
# Keep this list as the source of same-time process connections.
EDGES_75 = [
    # A) Direct control edges (valve ↔ measurement)
    ("xmv_3", "xmeas_1"),   # A feed flow valve ↔ A feed stream
    ("xmv_1", "xmeas_2"),   # D feed flow valve ↔ D feed stream
    ("xmv_2", "xmeas_3"),   # E feed flow valve ↔ E feed stream
    ("xmv_4", "xmeas_4"),   # Total feed flow to stripper valve ↔ Total fresh feed stripper
    ("xmv_5", "xmeas_5"),   # Compressor recycle valve ↔ Recycle flow into reactor
    ("xmv_6", "xmeas_10"),  # Purge valve ↔ Purge rate
    ("xmv_7", "xmeas_14"),  # Separator pot liquid flow valve ↔ Separator underflow
    ("xmv_8", "xmeas_17"),  # Stripper liquid product flow valve ↔ Stripper underflow
    ("xmv_9", "xmeas_19"),  # Stripper steam valve ↔ Stripper steam flow
    ("xmv_10", "xmeas_21"), # Reactor cooling water flow valve ↔ Reactor cooling water outlet temp
    ("xmv_11", "xmeas_22"), # Condenser cooling water flow valve ↔ Condenser cooling water outlet temp

    # B) Same-unit edges: Reactor bundle
    ("xmeas_6", "xmeas_7"),   # Reactor feed rate ↔ Reactor pressure
    ("xmeas_6", "xmeas_9"),   # Reactor feed rate ↔ Reactor temp
    ("xmeas_7", "xmeas_8"),   # Reactor pressure ↔ Reactor level
    ("xmeas_8", "xmeas_9"),   # Reactor level ↔ Reactor temp
    ("xmeas_9", "xmeas_21"),  # Reactor temp ↔ Reactor cooling water outlet temp
    ("xmv_10", "xmeas_9"),    # Reactor cooling valve ↔ Reactor temp
    ("xmeas_5", "xmeas_6"),   # Recycle flow ↔ Reactor feed rate
    ("xmeas_5", "xmeas_7"),   # Recycle flow ↔ Reactor pressure

    # B) Same-unit edges: Separator bundle
    ("xmeas_11", "xmeas_13"), # Separator temp ↔ Separator pressure
    ("xmeas_12", "xmeas_13"), # Separator level ↔ Separator pressure
    ("xmeas_12", "xmeas_14"), # Separator level ↔ Separator underflow
    ("xmeas_14", "xmeas_13"), # Separator underflow ↔ Separator pressure
    ("xmv_7", "xmeas_12"),    # Separator valve ↔ Separator level

    # B) Same-unit edges: Stripper bundle
    ("xmeas_15", "xmeas_16"), # Stripper level ↔ Stripper pressure
    ("xmeas_15", "xmeas_18"), # Stripper level ↔ Stripper temperature
    ("xmeas_16", "xmeas_18"), # Stripper pressure ↔ Stripper temperature
    ("xmeas_19", "xmeas_18"), # Stripper steam flow ↔ Stripper temperature
    ("xmv_9", "xmeas_18"),    # Stripper steam valve ↔ Stripper temperature
    ("xmeas_17", "xmeas_15"), # Stripper underflow ↔ Stripper level
    ("xmv_8", "xmeas_15"),    # Product valve ↔ Stripper level

    # B) Same-unit edges: Compressor / recycle loop bundle
    ("xmeas_20", "xmeas_5"),  # Compressor work ↔ Recycle flow
    ("xmeas_20", "xmeas_13"), # Compressor work ↔ Separator pressure
    ("xmeas_10", "xmeas_13"), # Purge rate ↔ Separator pressure

    # C) Stream connection edges: Feeds → Reactor
    ("xmeas_1", "xmeas_6"), # A feed stream ↔ Reactor feed rate
    ("xmeas_2", "xmeas_6"), # D feed stream ↔ Reactor feed rate
    ("xmeas_3", "xmeas_6"), # E feed stream ↔ Reactor feed rate

    # C) Reactor ↔ Separator
    ("xmeas_7", "xmeas_13"), # Reactor pressure ↔ Separator pressure
    ("xmeas_9", "xmeas_11"), # Reactor temp ↔ Separator temp

    # C) Condenser cooling ↔ Separator conditions
    ("xmeas_22", "xmeas_11"), # Condenser cooling water outlet temp ↔ Separator temp
    ("xmv_11", "xmeas_11"),   # Condenser cooling valve ↔ Separator temp

    # C) Separator bottoms → Stripper feed
    ("xmeas_14", "xmeas_15"), # Separator underflow ↔ Stripper level
    ("xmeas_14", "xmeas_17"), # Separator underflow ↔ Stripper underflow

    # C) Purge & recycle loop ties
    ("xmeas_10", "xmeas_5"), # Purge rate ↔ Recycle flow into reactor
    ("xmv_6", "xmeas_5"),    # Purge valve ↔ Recycle flow into reactor

    # D) Composition edges: Reactor feed composition ↔ Reactor feed/recycle
    ("xmeas_23", "xmeas_6"), # Comp A in reactor feed ↔ Reactor feed rate
    ("xmeas_24", "xmeas_6"), # Comp B in reactor feed ↔ Reactor feed rate
    ("xmeas_25", "xmeas_6"), # Comp C in reactor feed ↔ Reactor feed rate
    ("xmeas_26", "xmeas_6"), # Comp D in reactor feed ↔ Reactor feed rate
    ("xmeas_27", "xmeas_6"), # Comp E in reactor feed ↔ Reactor feed rate
    ("xmeas_28", "xmeas_6"), # Comp F in reactor feed ↔ Reactor feed rate

    ("xmeas_23", "xmeas_5"), # Comp A in reactor feed ↔ Recycle flow
    ("xmeas_24", "xmeas_5"), # Comp B in reactor feed ↔ Recycle flow
    ("xmeas_25", "xmeas_5"), # Comp C in reactor feed ↔ Recycle flow
    ("xmeas_26", "xmeas_5"), # Comp D in reactor feed ↔ Recycle flow
    ("xmeas_27", "xmeas_5"), # Comp E in reactor feed ↔ Recycle flow
    ("xmeas_28", "xmeas_5"), # Comp F in reactor feed ↔ Recycle flow

    # D) Composition edges: Purge composition ↔ Purge rate
    ("xmeas_29", "xmeas_10"), # Comp A in purge ↔ Purge rate
    ("xmeas_30", "xmeas_10"), # Comp B in purge ↔ Purge rate
    ("xmeas_31", "xmeas_10"), # Comp C in purge ↔ Purge rate
    ("xmeas_32", "xmeas_10"), # Comp D in purge ↔ Purge rate
    ("xmeas_33", "xmeas_10"), # Comp E in purge ↔ Purge rate
    ("xmeas_34", "xmeas_10"), # Comp F in purge ↔ Purge rate
    ("xmeas_35", "xmeas_10"), # Comp G in purge ↔ Purge rate
    ("xmeas_36", "xmeas_10"), # Comp H in purge ↔ Purge rate

    # D) Composition edges: Product composition ↔ Product flow + Stripper temperature
    ("xmeas_37", "xmeas_17"), # Comp D in product ↔ Stripper underflow/product flow
    ("xmeas_38", "xmeas_17"), # Comp E in product ↔ Stripper underflow/product flow
    ("xmeas_39", "xmeas_17"), # Comp F in product ↔ Stripper underflow/product flow
    ("xmeas_40", "xmeas_17"), # Comp G in product ↔ Stripper underflow/product flow
    ("xmeas_41", "xmeas_17"), # Comp H in product ↔ Stripper underflow/product flow

    ("xmeas_37", "xmeas_18"), # Comp D in product ↔ Stripper temperature
    ("xmeas_38", "xmeas_18"), # Comp E in product ↔ Stripper temperature
    ("xmeas_39", "xmeas_18"), # Comp F in product ↔ Stripper temperature
    ("xmeas_40", "xmeas_18"), # Comp G in product ↔ Stripper temperature
    ("xmeas_41", "xmeas_18"), # Comp H in product ↔ Stripper temperature
]


def get_sensor_names():
    sensor_names = []

    sensor_index = 1
    while sensor_index <= 41:
        sensor_names.append(f"xmeas_{sensor_index}")
        sensor_index += 1

    sensor_index = 1
    while sensor_index <= 11:
        sensor_names.append(f"xmv_{sensor_index}")
        sensor_index += 1

    return sensor_names


def make_sensor_name_to_index():
    sensor_names = get_sensor_names()
    sensor_name_to_index = {}

    index = 0
    while index < len(sensor_names):
        sensor_name = sensor_names[index]
        sensor_name_to_index[sensor_name] = index
        index += 1

    return sensor_name_to_index


def make_spatiotemporal_node_id(time_step, sensor_index, num_sensors):
    node_id = (time_step * num_sensors) + sensor_index
    return node_id


def build_process_sensor_edges():
    sensor_name_to_index = make_sensor_name_to_index()
    process_sensor_edges = []

    edge_index = 0
    while edge_index < len(EDGES_75):
        source_name, target_name = EDGES_75[edge_index]

        if source_name not in sensor_name_to_index:
            raise KeyError(f"Unknown source sensor name in edge list: {source_name}")
        if target_name not in sensor_name_to_index:
            raise KeyError(f"Unknown target sensor name in edge list: {target_name}")

        source_sensor_index = sensor_name_to_index[source_name]
        target_sensor_index = sensor_name_to_index[target_name]
        process_sensor_edges.append((source_sensor_index, target_sensor_index))
        edge_index += 1

    return process_sensor_edges


def build_spatiotemporal_edge_index(window_size, num_sensors):
    process_sensor_edges = build_process_sensor_edges()
    edge_pairs = []

    # Temporal edges: same sensor, forward in time only
    time_step = 0
    while time_step < window_size - 1:
        sensor_index = 0
        while sensor_index < num_sensors:
            current_node_id = make_spatiotemporal_node_id(time_step, sensor_index, num_sensors)
            next_node_id = make_spatiotemporal_node_id(time_step + 1, sensor_index, num_sensors)
            edge_pairs.append([current_node_id, next_node_id])
            sensor_index += 1

        time_step += 1

    # Process edges: same time step, undirected
    time_step = 0
    while time_step < window_size:
        process_edge_index = 0
        while process_edge_index < len(process_sensor_edges):
            source_sensor_index, target_sensor_index = process_sensor_edges[process_edge_index]

            source_node_id = make_spatiotemporal_node_id(time_step, source_sensor_index, num_sensors)
            target_node_id = make_spatiotemporal_node_id(time_step, target_sensor_index, num_sensors)

            edge_pairs.append([source_node_id, target_node_id])
            edge_pairs.append([target_node_id, source_node_id])
            process_edge_index += 1

        time_step += 1

    edge_index = torch.tensor(edge_pairs, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    return edge_index


def prepare_spatiotemporal_node_features(inputs_2d):
    # One scalar feature per (sensor, time) node.
    inputs_np = np.asarray(inputs_2d, dtype=np.float32)
    num_examples = inputs_np.shape[0]
    graph_inputs_np = inputs_np.reshape(num_examples, -1, 1).copy()
    graph_inputs_tensor = torch.tensor(graph_inputs_np, dtype=torch.float32)
    return graph_inputs_tensor


class TEPWindowGraphDataset:
    def __init__(self, graph_node_features, labels, edge_index):
        self.graph_node_features = graph_node_features
        self.labels = labels
        self.edge_index = edge_index

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.graph_node_features[index]
        y = self.labels[index]
        graph_data = Data(x=x, edge_index=self.edge_index, y=y)
        return graph_data


class TEPGNN(nn.Module):
    def __init__(self, num_classes, hidden_dim=64):
        super().__init__()

        self.gnn1 = SAGEConv(1, hidden_dim)
        self.relu1 = nn.ReLU()

        self.gnn2 = SAGEConv(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.gnn1(x, edge_index)
        x = self.relu1(x)

        x = self.gnn2(x, edge_index)
        x = self.relu2(x)

        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x
