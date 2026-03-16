import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from models.simplemlp import SimpleMLP
import torch.nn as nn


DATA_FILE_NAME = "small_fault0_20_runs1_500_W60_step10.npz"

def absolute_path_of_npz_data():
    file_name = DATA_FILE_NAME

    # finding npz data file
    current_file_path = Path(__file__).resolve()
    current_directory = current_file_path.parent
    parent_directory = current_directory.parent
    parent_directory = os.path.abspath(parent_directory)
    data_directory = os.path.join(parent_directory, "data")
    # raw_data_directory = os.path.join(data_directory, "raw")
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
            # then this run is a validation run
            validation_inputs.append(inputs[i])
            validation_answers.append(answers[i])
            i += 1
            continue
        # else this run is a training run
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
    # IMPORTANT:
    # compute normalization stats from TRAIN only
    # so validation remains truly unseen during preprocessing
    mean = np.mean(training_inputs, axis=0)
    std = np.std(training_inputs, axis=0)

    # avoid dividing by zero
    std = std + 1e-8

    train_inputs = (training_inputs - mean) / std
    validate_inputs = (validation_inputs - mean) / std

    return train_inputs, validate_inputs, mean, std

def make_loaders(training_inputs, training_answers, validation_inputs, validation_answers, batch_size):
    train_in = torch.tensor(training_inputs, dtype=torch.float32)
    train_ans = torch.tensor(training_answers, dtype=torch.long)

    val_in = torch.tensor(validation_inputs, dtype=torch.float32)
    val_ans = torch.tensor(validation_answers, dtype=torch.long)

    train_ds = TensorDataset(train_in, train_ans)
    val_ds = TensorDataset(val_in, val_ans)

    # shuffle training data
    # no need to shuffle validation data
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader




def pick_device():
    # Use Mac GPU if available, else CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# training for each epoch (full pass through all training examples)
def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    # Put model in training mode (turns on training behaviors)
    model.train()

    total_loss = 0.0        # total summed loss across all examples
    total_correct = 0       # number of correct predictions
    total_examples = 0      # how many examples we’ve processed

    # Loop through batches of (inputs, answers)
    for batch_inputs, batch_answers in train_loader:
        # Move data to device (mps/cpu)
        # Model + data must be on the same device.
        batch_inputs = batch_inputs.to(device)
        batch_answers = batch_answers.to(device)

        # PyTorch accumulates gradients by default.
        # Clear old gradients from last step
        optimizer.zero_grad()

        # feed inputs through network, get output from last layer
        # Forward pass: model outputs scores for each class
        logits = model(batch_inputs)

        # loss_fn will be torch.nn.CrossEntropyLoss
        # Loss: how wrong the scores are compared to the true answers
        # returns one number: the batch’s average loss
        loss = loss_fn(logits, batch_answers)

        # How should each weight change to make the loss smaller?
        # like that minimizing the function visual in 3Blue1Brown video
        # which way should the ball roll down and at what slope
        # Backward pass: compute gradients
        loss.backward()

        # optimizer uses those gradients to adjust the model’s weights
        # Update weights using gradients
        optimizer.step()

        # ---- Metrics tracking ----
        batch_size = batch_answers.shape[0]  # Usually 64, except possibly the last batch
        total_loss = total_loss + (loss.item() * batch_size) # Add average loss for the batch times number of batches

        # Convert logits to predicted classes
        predictions = torch.argmax(logits, dim=1)

        # Count how many predictions are correct
        correct_in_batch = (predictions == batch_answers).sum().item()

        # Update running totals for the whole epoch
        total_correct = total_correct + correct_in_batch
        total_examples = total_examples + batch_size

    # average loss per example for the whole epoch
    # accuracy for the whole epoch
    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples

    return avg_loss, avg_acc



# evaluation on the validation dataset
def validate_one_epoch(model, val_loader, loss_fn, device):
    # Put model in evaluation mode (turns off training behaviors like dropout)
    model.eval()

    total_loss = 0.0        # total summed loss across all examples
    total_correct = 0       # number of correct predictions
    total_examples = 0      # how many examples we’ve processed

    # No gradients needed for validation
    with torch.no_grad():
        for batch_inputs, batch_answers in val_loader:
            batch_inputs = batch_inputs.to(device)
            batch_answers = batch_answers.to(device)

            # feed inputs through network, get output from last layer
            logits = model(batch_inputs)

            # Loss: how wrong the scores are compared to the true answers
            # returns one number: the batch’s average loss
            loss = loss_fn(logits, batch_answers)


            batch_size = batch_answers.shape[0]
            total_loss = total_loss + (loss.item() * batch_size) # Add average loss for the batch times number of batches

            # Convert logits to predicted classes
            predictions = torch.argmax(logits, dim=1)

            # Count how many predictions are correct
            correct_in_batch = (predictions == batch_answers).sum().item()

            # Update running totals for the whole epoch
            total_correct = total_correct + correct_in_batch
            total_examples = total_examples + batch_size

    # average loss per example for the whole validation epoch
    # accuracy for the whole epoch
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

    # split into training and validation sets
    # split by simulationRun
    # the validation set exists so that we can test on that instead of actual testing data
    # because actual testing data is reserved for final score of model
    validation_runs = list(range(376, 501)) # 16-20
    training_set, validation_set = split_by_runs(inputs, answers, run_ids, validation_runs)

    inputs_training_np = training_set[0]
    answers_training_np = training_set[1]
    inputs_validation_np = validation_set[0]
    answers_validation_np = validation_set[1]

    print("training set shape --- inputs:", inputs_training_np.shape, "answers:", answers_training_np.shape)
    print("validation set shape --- inputs:", inputs_validation_np.shape, "answers:", answers_validation_np.shape)

    # normalize inputs using TRAIN stats only
    inputs_training_np, inputs_validation_np, normalization_mean, normalization_std = normalize_train_and_val(
        training_inputs=inputs_training_np,
        validation_inputs=inputs_validation_np
    )

    print("training min/max:", float(np.min(inputs_training_np)), float(np.max(inputs_training_np)))
    print("validation min/max:", float(np.min(inputs_validation_np)), float(np.max(inputs_validation_np)))


    # convert numpy arrays to PyTorch tensors
    training_loader, validation_loader = make_loaders(
        training_inputs=inputs_training_np,
        training_answers = answers_training_np,
        validation_inputs=inputs_validation_np,
        validation_answers=answers_validation_np,
        batch_size=64
    )
    print("num train batches:", len(training_loader))
    print("num val batches:", len(validation_loader))


    # now build training loop that teaches the network
    device = pick_device()
    print("Using device:", device)

    class_labels = np.unique(answers).astype(np.int64)
    num_classes = len(class_labels)
    input_dim = int(inputs.shape[1])
    model = SimpleMLP(input_dim=input_dim, num_classes=num_classes)
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

    # save trained model checkpoint
    # checkpoint contract:
    # - model weights
    # - model dimensions
    # - class label order
    # - normalization values used during training
    # - validation split and data metadata
    save_path = "baseline_mlp.pt"
    # Save metadata as tensors / plain Python values
    # so checkpoints are friendly with torch.load(weights_only=True).
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
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
