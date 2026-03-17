import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from models.cnn1d import SimpleCNN1D
import torch.nn as nn


DATA_FILE_NAME = "small_fault0_20_runs1_500_W60_step10_2d.npz"


def absolute_path_of_npz_data():
    file_name = DATA_FILE_NAME

    # finding npz data file
    current_file_path = Path(__file__).resolve()
    current_directory = current_file_path.parent
    parent_directory = current_directory.parent
    parent_directory = os.path.abspath(parent_directory)
    data_directory = os.path.join(parent_directory, "data")
    processed_data_directory = os.path.join(data_directory, "processed")

    npz_file_path = os.path.join(
        processed_data_directory,
        file_name
    )

    return npz_file_path


def split_by_runs(inputs, answers, run_ids, validation_runs):
    training_inputs = []
    training_answers = []
    validation_inputs = []
    validation_answers = []

    i = 0
    while i < len(answers):
        run = int(run_ids[i])
        if run in validation_runs:
            validation_inputs.append(inputs[i])
            validation_answers.append(answers[i])
            i += 1
            continue

        training_inputs.append(inputs[i])
        training_answers.append(answers[i])
        i += 1

    inputs_training_np = np.array(training_inputs, dtype=np.float32)
    answers_training_np = np.array(training_answers, dtype=np.int8)
    training_set = [inputs_training_np, answers_training_np]

    inputs_validation_np = np.array(validation_inputs, dtype=np.float32)
    answers_validation_np = np.array(validation_answers, dtype=np.int8)
    validation_set = [inputs_validation_np, answers_validation_np]

    return training_set, validation_set


def normalize_train_and_val(training_inputs, validation_inputs):
    # For 2D time-series data:
    # shape is (num_examples, window_size, num_sensors)
    #
    # Compute one mean/std per sensor channel using TRAIN only.
    # We average across:
    # - all training examples
    # - all time steps
    #
    # Result shape is (52,)
    mean = np.mean(training_inputs, axis=(0, 1))
    std = np.std(training_inputs, axis=(0, 1))

    # avoid dividing by zero
    std = std + 1e-8

    train_inputs = (training_inputs - mean) / std
    validate_inputs = (validation_inputs - mean) / std

    return train_inputs, validate_inputs, mean, std


def transpose_for_conv1d(inputs):
    # Current shape is (num_examples, window_size, num_sensors)
    # Conv1d expects (num_examples, num_sensors, window_size)
    return np.transpose(inputs, (0, 2, 1))


def make_loaders(training_inputs, training_answers, validation_inputs, validation_answers, batch_size):
    train_in = torch.tensor(training_inputs, dtype=torch.float32)
    train_ans = torch.tensor(training_answers, dtype=torch.long)

    val_in = torch.tensor(validation_inputs, dtype=torch.float32)
    val_ans = torch.tensor(validation_answers, dtype=torch.long)

    train_ds = TensorDataset(train_in, train_ans)
    val_ds = TensorDataset(val_in, val_ans)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def pick_device():
    # Use Mac GPU if available, else CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_inputs, batch_answers in train_loader:
        batch_inputs = batch_inputs.to(device)
        batch_answers = batch_answers.to(device)

        optimizer.zero_grad()

        logits = model(batch_inputs)
        loss = loss_fn(logits, batch_answers)

        loss.backward()
        optimizer.step()

        batch_size = batch_answers.shape[0]
        total_loss = total_loss + (loss.item() * batch_size)

        predictions = torch.argmax(logits, dim=1)
        correct_in_batch = (predictions == batch_answers).sum().item()

        total_correct = total_correct + correct_in_batch
        total_examples = total_examples + batch_size

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples

    return avg_loss, avg_acc


def validate_one_epoch(model, val_loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch_inputs, batch_answers in val_loader:
            batch_inputs = batch_inputs.to(device)
            batch_answers = batch_answers.to(device)

            logits = model(batch_inputs)
            loss = loss_fn(logits, batch_answers)

            batch_size = batch_answers.shape[0]
            total_loss = total_loss + (loss.item() * batch_size)

            predictions = torch.argmax(logits, dim=1)
            correct_in_batch = (predictions == batch_answers).sum().item()

            total_correct = total_correct + correct_in_batch
            total_examples = total_examples + batch_size

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples

    return avg_loss, avg_acc


def main():
    # load dataset
    file_path = absolute_path_of_npz_data()
    data = np.load(file_path, allow_pickle=True)

    inputs = data["inputs"]
    answers = data["answers"]
    run_ids = data["run_ids"]
    window_size = int(data["window_size"]) if "window_size" in data else 60
    step_size = int(data["step_size"]) if "step_size" in data else 10

    data.close()

    print("inputs shape:", inputs.shape)
    print("answers shape:", answers.shape)
    print("run_ids shape:", run_ids.shape)

    validation_runs = list(range(376, 501))
    training_set, validation_set = split_by_runs(inputs, answers, run_ids, validation_runs)

    inputs_training_np = training_set[0]
    answers_training_np = training_set[1]
    inputs_validation_np = validation_set[0]
    answers_validation_np = validation_set[1]

    print("training set shape --- inputs:", inputs_training_np.shape, "answers:", answers_training_np.shape)
    print("validation set shape --- inputs:", inputs_validation_np.shape, "answers:", answers_validation_np.shape)

    # normalize using TRAIN only
    inputs_training_np, inputs_validation_np, normalization_mean, normalization_std = normalize_train_and_val(
        training_inputs=inputs_training_np,
        validation_inputs=inputs_validation_np
    )

    print("training min/max:", float(np.min(inputs_training_np)), float(np.max(inputs_training_np)))
    print("validation min/max:", float(np.min(inputs_validation_np)), float(np.max(inputs_validation_np)))

    # transpose to channel-first format for Conv1d
    inputs_training_np = transpose_for_conv1d(inputs_training_np)
    inputs_validation_np = transpose_for_conv1d(inputs_validation_np)

    print("training shape for cnn:", inputs_training_np.shape)
    print("validation shape for cnn:", inputs_validation_np.shape)

    training_loader, validation_loader = make_loaders(
        training_inputs=inputs_training_np,
        training_answers=answers_training_np,
        validation_inputs=inputs_validation_np,
        validation_answers=answers_validation_np,
        batch_size=64
    )
    print("num train batches:", len(training_loader))
    print("num val batches:", len(validation_loader))

    device = pick_device()
    print("Using device:", device)

    class_labels = np.unique(answers).astype(np.int64)
    num_classes = len(class_labels)
    num_channels = int(inputs_training_np.shape[1])
    sequence_length = int(inputs_training_np.shape[2])

    model = SimpleCNN1D(num_channels=num_channels, num_classes=num_classes)
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    start_time = time.time()

    epochs = 10
    epoch = 1

    while epoch <= epochs:
        train_loss, train_acc = train_one_epoch(model, training_loader, loss_function, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, validation_loader, loss_function, device)

        print(
            "Epoch", epoch,
            "| train loss", round(train_loss, 4), "acc", f"{(train_acc * 100):.1f}%",
            "| val loss", round(val_loss, 4), "acc", f"{(val_acc * 100):.1f}%"
        )

        epoch += 1

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print("Training / Validation Loop --- Time elapsed:", minutes, "min", seconds, "sec")

    save_path = "cnn1d_baseline.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_channels": num_channels,
        "sequence_length": sequence_length,
        "num_classes": num_classes,
        "class_labels": torch.tensor(class_labels.tolist(), dtype=torch.long),
        "normalization_mean": torch.tensor(normalization_mean, dtype=torch.float32),
        "normalization_std": torch.tensor(normalization_std, dtype=torch.float32),
        "validation_runs": torch.tensor(validation_runs, dtype=torch.int32),
        "data_file_name": DATA_FILE_NAME,
        "window_size": window_size,
        "step_size": step_size,
        "epochs": epochs,
    }
    torch.save(checkpoint, save_path)

    print("Saved trained model to:", save_path)


if __name__ == "__main__":
    main()
