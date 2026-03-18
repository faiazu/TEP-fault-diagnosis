import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn

from models.gnn_model import (
    TEPGNN,
    TEPWindowGraphDataset,
    build_spatiotemporal_edge_index,
    prepare_spatiotemporal_node_features,
)

try:
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ModuleNotFoundError:
    PyGDataLoader = None


DATA_FILE_NAME = "small_fault0_20_runs1_20_W60_step10_2d.npz"


def absolute_path_of_npz_data():
    current_file_path = Path(__file__).resolve()
    current_directory = current_file_path.parent
    parent_directory = current_directory.parent
    parent_directory = os.path.abspath(parent_directory)
    data_directory = os.path.join(parent_directory, "data")
    processed_data_directory = os.path.join(data_directory, "processed")

    npz_file_path = os.path.join(
        processed_data_directory,
        DATA_FILE_NAME
    )

    return npz_file_path


def split_by_runs(inputs, answers, run_ids, validation_runs):
    training_inputs = []
    training_answers = []
    validation_inputs = []
    validation_answers = []

    index = 0
    while index < len(answers):
        run_id = int(run_ids[index])

        if run_id in validation_runs:
            validation_inputs.append(inputs[index])
            validation_answers.append(answers[index])
        else:
            training_inputs.append(inputs[index])
            training_answers.append(answers[index])

        index += 1

    inputs_training_np = np.array(training_inputs, dtype=np.float32)
    answers_training_np = np.array(training_answers, dtype=np.int64)

    inputs_validation_np = np.array(validation_inputs, dtype=np.float32)
    answers_validation_np = np.array(validation_answers, dtype=np.int64)

    training_set = [inputs_training_np, answers_training_np]
    validation_set = [inputs_validation_np, answers_validation_np]
    return training_set, validation_set


def choose_validation_runs(run_ids):
    unique_run_ids = sorted(np.unique(run_ids).astype(int).tolist())

    preferred_validation_runs = list(range(376, 501))
    available_preferred_runs = []

    index = 0
    while index < len(preferred_validation_runs):
        run_id = preferred_validation_runs[index]
        if run_id in unique_run_ids:
            available_preferred_runs.append(run_id)
        index += 1

    if len(available_preferred_runs) > 0:
        return available_preferred_runs

    fallback_count = max(1, len(unique_run_ids) // 4)
    fallback_validation_runs = unique_run_ids[-fallback_count:]
    return fallback_validation_runs


def normalize_train_and_val(training_inputs, validation_inputs):
    mean = np.mean(training_inputs, axis=(0, 1))
    std = np.std(training_inputs, axis=(0, 1))
    std = std + 1e-8

    train_inputs = (training_inputs - mean) / std
    validate_inputs = (validation_inputs - mean) / std

    return train_inputs, validate_inputs, mean, std


def prepare_graph_labels(labels):
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return labels_tensor


def make_graph_loaders(training_graphs, training_labels, validation_graphs, validation_labels, edge_index, batch_size):
    num_workers = 4
    persistent_workers = True

    training_dataset = TEPWindowGraphDataset(
        graph_node_features=training_graphs,
        labels=training_labels,
        edge_index=edge_index
    )
    validation_dataset = TEPWindowGraphDataset(
        graph_node_features=validation_graphs,
        labels=validation_labels,
        edge_index=edge_index
    )

    training_loader = PyGDataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers
    )
    validation_loader = PyGDataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers
    )

    return training_dataset, validation_dataset, training_loader, validation_loader


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    num_batches = len(train_loader)
    print_every_batches = 100

    batch_index = 0
    for batch_data in train_loader:
        batch_index += 1
        batch_data = batch_data.to(device)

        optimizer.zero_grad()

        logits = model(batch_data)
        loss = loss_fn(logits, batch_data.y)

        loss.backward()
        optimizer.step()

        batch_size = batch_data.y.shape[0]
        total_loss = total_loss + (loss.item() * batch_size)

        predictions = torch.argmax(logits, dim=1)
        correct_in_batch = (predictions == batch_data.y).sum().item()

        total_correct = total_correct + correct_in_batch
        total_examples = total_examples + batch_size

        if batch_index == 1 or batch_index % print_every_batches == 0 or batch_index == num_batches:
            running_loss = total_loss / total_examples
            running_acc = total_correct / total_examples
            print(
                "  train batch", batch_index, "/", num_batches,
                "| running loss", round(running_loss, 4),
                "| running acc", f"{(running_acc * 100):.1f}%"
            )

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc


def validate_one_epoch(model, val_loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    num_batches = len(val_loader)
    print_every_batches = 100

    batch_index = 0
    with torch.no_grad():
        for batch_data in val_loader:
            batch_index += 1
            batch_data = batch_data.to(device)

            logits = model(batch_data)
            loss = loss_fn(logits, batch_data.y)

            batch_size = batch_data.y.shape[0]
            total_loss = total_loss + (loss.item() * batch_size)

            predictions = torch.argmax(logits, dim=1)
            correct_in_batch = (predictions == batch_data.y).sum().item()

            total_correct = total_correct + correct_in_batch
            total_examples = total_examples + batch_size

            if batch_index == 1 or batch_index % print_every_batches == 0 or batch_index == num_batches:
                running_loss = total_loss / total_examples
                running_acc = total_correct / total_examples
                print(
                    "  val batch", batch_index, "/", num_batches,
                    "| running loss", round(running_loss, 4),
                    "| running acc", f"{(running_acc * 100):.1f}%"
                )

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc


def main():
    if PyGDataLoader is None:
        raise ImportError("torch_geometric is required to run train_gnn.py")

    file_path = absolute_path_of_npz_data()
    data = np.load(file_path, allow_pickle=True)

    if "inputs" in data:
        inputs = data["inputs"]
    elif "inputs_2d" in data:
        inputs = data["inputs_2d"]
    else:
        data.close()
        raise KeyError("Dataset must contain inputs or inputs_2d.")

    if "answers" in data:
        answers = data["answers"]
    elif "labels" in data:
        answers = data["labels"]
    else:
        data.close()
        raise KeyError("Dataset must contain answers or labels.")

    run_ids = data["run_ids"]
    window_size = int(data["window_size"]) if "window_size" in data else 60
    step_size = int(data["step_size"]) if "step_size" in data else 10
    num_sensors = int(inputs.shape[2])
    data.close()

    print("Loaded dataset shapes:")
    print("inputs shape:", inputs.shape)
    print("answers shape:", answers.shape)
    print("run_ids shape:", run_ids.shape)

    validation_runs = choose_validation_runs(run_ids)
    print("validation runs:", validation_runs)

    training_set, validation_set = split_by_runs(inputs, answers, run_ids, validation_runs)

    inputs_training_np = training_set[0]
    answers_training_np = training_set[1]
    inputs_validation_np = validation_set[0]
    answers_validation_np = validation_set[1]

    print("training set shape --- inputs:", inputs_training_np.shape, "answers:", answers_training_np.shape)
    print("validation set shape --- inputs:", inputs_validation_np.shape, "answers:", answers_validation_np.shape)

    if len(inputs_training_np) == 0:
        raise ValueError("Training split is empty.")
    if len(inputs_validation_np) == 0:
        raise ValueError("Validation split is empty. Check the validation run selection.")

    inputs_training_np, inputs_validation_np, train_mean, train_std = normalize_train_and_val(
        training_inputs=inputs_training_np,
        validation_inputs=inputs_validation_np
    )

    print("training min/max:", float(np.min(inputs_training_np)), float(np.max(inputs_training_np)))
    print("validation min/max:", float(np.min(inputs_validation_np)), float(np.max(inputs_validation_np)))

    training_graphs = prepare_spatiotemporal_node_features(inputs_training_np)
    validation_graphs = prepare_spatiotemporal_node_features(inputs_validation_np)
    training_labels = prepare_graph_labels(answers_training_np)
    validation_labels = prepare_graph_labels(answers_validation_np)

    print("graph-ready training shape:", tuple(training_graphs.shape))
    print("graph-ready validation shape:", tuple(validation_graphs.shape))

    edge_index = build_spatiotemporal_edge_index(
        window_size=window_size,
        num_sensors=num_sensors
    )
    print("edge_index shape:", tuple(edge_index.shape))
    print("number of directed edges in edge_index:", int(edge_index.shape[1]))

    training_dataset, validation_dataset, training_loader, validation_loader = make_graph_loaders(
        training_graphs=training_graphs,
        training_labels=training_labels,
        validation_graphs=validation_graphs,
        validation_labels=validation_labels,
        edge_index=edge_index,
        batch_size=16
    )

    sample_graph = training_dataset[0]
    print("One sample node feature shape:", tuple(sample_graph.x.shape))

    print("num train batches:", len(training_loader))
    print("num val batches:", len(validation_loader))

    device = pick_device()
    print("Using device:", device)

    class_labels = np.unique(answers).astype(np.int64)
    num_classes = len(class_labels)
    hidden_dim = 64

    model = TEPGNN(
        num_classes=num_classes,
        hidden_dim=hidden_dim
    )
    model = model.to(device)

    example_batch = next(iter(training_loader))
    example_batch = example_batch.to(device)
    with torch.no_grad():
        example_output = model(example_batch)
    print("One batch output shape:", tuple(example_output.shape))

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    start_time = time.time()

    epochs = 10
    epoch = 1

    while epoch <= epochs:
        epoch_start_time = time.time()

        print("\nStarting epoch", epoch)
        train_loss, train_acc = train_one_epoch(model, training_loader, loss_function, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, validation_loader, loss_function, device)

        epoch_elapsed = time.time() - epoch_start_time
        epoch_minutes = int(epoch_elapsed // 60)
        epoch_seconds = int(epoch_elapsed % 60)

        print(
            "Epoch", epoch,
            "| train loss", round(train_loss, 4), "acc", f"{(train_acc * 100):.1f}%",
            "| val loss", round(val_loss, 4), "acc", f"{(val_acc * 100):.1f}%",
            "| epoch time", str(epoch_minutes) + "m", str(epoch_seconds) + "s"
        )

        epoch += 1

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print("Training / Validation Loop --- Time elapsed:", minutes, "min", seconds, "sec")

    save_path = "gnn_model.pt"
    checkpoint = {
        "model_type": "gnn",
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "class_labels": torch.tensor(class_labels.tolist(), dtype=torch.long),
        "hidden_dim": hidden_dim,
        "num_sensors": num_sensors,
        "window_size": window_size,
        "step_size": step_size,
        "dataset_path": file_path,
        "data_file_name": DATA_FILE_NAME,
        "validation_runs": torch.tensor(validation_runs, dtype=torch.int32),
        "train_mean": torch.tensor(train_mean, dtype=torch.float32),
        "train_std": torch.tensor(train_std, dtype=torch.float32),
        "normalization_mean": torch.tensor(train_mean, dtype=torch.float32),
        "normalization_std": torch.tensor(train_std, dtype=torch.float32),
        "edge_list_version": "process_edges_75",
        "process_edges_are_undirected": True,
        "epochs": epochs,
    }
    torch.save(checkpoint, save_path)

    print("Saved trained model to:", save_path)


if __name__ == "__main__":
    main()
