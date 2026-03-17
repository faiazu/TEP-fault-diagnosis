from pathlib import Path
import argparse
import csv
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from models.cnn1d import SimpleCNN1D
from models.simplemlp import SimpleMLP



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["mlp", "cnn"],
        default="mlp",
        help="Which saved model to evaluate."
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    parent_directory = os.path.abspath(project_root)
    data_directory = os.path.join(parent_directory, "data")
    processed_data_directory = os.path.join(data_directory, "processed")

    cnn_checkpoint_path = os.path.join(parent_directory, "cnn1d_baseline.pt")
    mlp_checkpoint_path = os.path.join(parent_directory, "baseline_mlp.pt")

    if args.model == "cnn":
        model_file_path = cnn_checkpoint_path
    else:
        model_file_path = mlp_checkpoint_path

    print("Chosen model type:", args.model)


    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    if not os.path.exists(model_file_path):
        raise FileNotFoundError("Checkpoint file not found: " + model_file_path)

    checkpoint = torch.load(
        model_file_path,
        map_location=device,
        weights_only=True
    )

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary.")

    required_checkpoint_keys = [
        "model_state_dict",
        "num_classes",
        "class_labels",
        "normalization_mean",
        "normalization_std",
        "validation_runs",
        "data_file_name",
    ]
    missing_keys = [key for key in required_checkpoint_keys if key not in checkpoint]
    if len(missing_keys) > 0:
        raise KeyError("Checkpoint is missing required keys: " + str(missing_keys))

    print("Loaded checkpoint:", model_file_path)
    print("Checkpoint keys:", sorted(checkpoint.keys()))
    

    num_classes = int(checkpoint["num_classes"])
    model_type = None

    if "input_dim" in checkpoint:
        model_type = "mlp"
        input_dim = int(checkpoint["input_dim"])
        model = SimpleMLP(input_dim=input_dim, num_classes=num_classes)
        print("Model type: simple mlp")
        print("Model input_dim:", input_dim)
    elif "num_channels" in checkpoint and "sequence_length" in checkpoint:
        model_type = "cnn1d"
        num_channels = int(checkpoint["num_channels"])
        sequence_length = int(checkpoint["sequence_length"])
        model = SimpleCNN1D(num_channels=num_channels, num_classes=num_classes)
        print("Model type: simple cnn1d")
        print("Model num_channels:", num_channels)
        print("Model sequence_length:", sequence_length)
    else:
        raise KeyError(
            "Checkpoint does not contain enough model shape information. "
            "Expected either input_dim for MLP, or num_channels and sequence_length for CNN."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print("Model loaded.")
    print("Model num_classes:", num_classes)

    data_file_path = os.path.join(processed_data_directory, checkpoint["data_file_name"])
    if not os.path.exists(data_file_path):
        raise FileNotFoundError("Evaluation dataset file not found: " + data_file_path)

    data = np.load(data_file_path, allow_pickle=True)

    required_data_keys = ["inputs", "answers", "run_ids"]
    missing_data_keys = [key for key in required_data_keys if key not in data]
    if len(missing_data_keys) > 0:
        data.close()
        raise KeyError("Dataset is missing required keys: " + str(missing_data_keys))

    inputs = data["inputs"]
    answers = data["answers"]
    run_ids = data["run_ids"]
    data.close()

    print("Loaded evaluation dataset:", data_file_path)
    print("inputs shape:", inputs.shape)
    print("answers shape:", answers.shape)
    print("run_ids shape:", run_ids.shape)

    split_mode = "validation_runs"

    if split_mode == "full":
        inputs_selected = inputs
        answers_selected = answers
        run_ids_selected = run_ids
    elif split_mode == "validation_runs":
        validation_runs = checkpoint["validation_runs"].detach().cpu().numpy()
        keep_mask = np.isin(run_ids, validation_runs)

        inputs_selected = inputs[keep_mask]
        answers_selected = answers[keep_mask]
        run_ids_selected = run_ids[keep_mask]
    else:
        raise ValueError("Unsupported split_mode: " + str(split_mode))

    print("Split mode:", split_mode)
    print("Selected examples:", len(answers_selected))
    print("Selected run ID range:", int(np.min(run_ids_selected)), "to", int(np.max(run_ids_selected)))

    normalization_mean = checkpoint["normalization_mean"].detach().cpu().numpy()
    normalization_std = checkpoint["normalization_std"].detach().cpu().numpy()

    if model_type == "mlp":
        if normalization_mean.shape[0] != inputs_selected.shape[1]:
            raise ValueError(
                "normalization_mean length does not match input dimension. "
                f"mean length={normalization_mean.shape[0]}, input_dim={inputs_selected.shape[1]}"
            )
        if normalization_std.shape[0] != inputs_selected.shape[1]:
            raise ValueError(
                "normalization_std length does not match input dimension. "
                f"std length={normalization_std.shape[0]}, input_dim={inputs_selected.shape[1]}"
            )
    elif model_type == "cnn1d":
        if inputs_selected.ndim != 3:
            raise ValueError(f"CNN expects 3D inputs (N, window_size, num_sensors), got {inputs_selected.shape}")
        if normalization_mean.shape[0] != inputs_selected.shape[2]:
            raise ValueError(
                "normalization_mean length does not match number of sensor channels. "
                f"mean length={normalization_mean.shape[0]}, num_sensors={inputs_selected.shape[2]}"
            )
        if normalization_std.shape[0] != inputs_selected.shape[2]:
            raise ValueError(
                "normalization_std length does not match number of sensor channels. "
                f"std length={normalization_std.shape[0]}, num_sensors={inputs_selected.shape[2]}"
            )

    inputs_norm = (inputs_selected - normalization_mean) / normalization_std

    if model_type == "cnn1d":
        inputs_norm = np.transpose(inputs_norm, (0, 2, 1))

    print("Normalization applied using checkpoint stats.")
    print("Normalized input min/max:", float(np.min(inputs_norm)), float(np.max(inputs_norm)))

    batch_size = 64

    inputs_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
    answers_tensor = torch.tensor(answers_selected, dtype=torch.long)

    evaluation_dataset = TensorDataset(inputs_tensor, answers_tensor)
    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print("Evaluation batch size:", batch_size)
    print("Evaluation batches:", len(evaluation_loader))

    all_logits = []
    all_true_labels = []

    with torch.no_grad():
        for batch_inputs, batch_answers in evaluation_loader:
            batch_inputs = batch_inputs.to(device)
            batch_answers = batch_answers.to(device)

            batch_logits = model(batch_inputs)

            all_logits.append(batch_logits.cpu())
            all_true_labels.append(batch_answers.cpu())

    if len(all_logits) == 0:
        raise ValueError("No batches found in evaluation loader.")

    logits_tensor = torch.cat(all_logits, dim=0)
    true_labels_tensor = torch.cat(all_true_labels, dim=0)

    logits_np = logits_tensor.numpy()
    true_labels_np = true_labels_tensor.numpy()

    print("Inference done.")
    print("logits shape:", logits_np.shape)
    print("true labels shape:", true_labels_np.shape)

    num_classes = logits_np.shape[1]
    total_examples = logits_np.shape[0]

    topk_accuracies = {}
    k_values = [1, 2, 3]
    sorted_class_indices = np.argsort(logits_np, axis=1)
    sorted_class_indices = sorted_class_indices[:, ::-1]

    for k in k_values:
        topk_class_indices = sorted_class_indices[:, :k]

        correct_predictions = 0
        for i in range(total_examples):
            true_class = true_labels_np[i]
            predicted_topk_labels = topk_class_indices[i]

            if true_class in predicted_topk_labels:
                correct_predictions += 1

        topk_accuracy = correct_predictions / total_examples
        topk_accuracies[k] = topk_accuracy

    print("Top-1 accuracy:", round(topk_accuracies[1] * 100, 2), "%")
    print("Top-2 accuracy:", round(topk_accuracies[2] * 100, 2), "%")
    print("Top-3 accuracy:", round(topk_accuracies[3] * 100, 2), "%")

    predicted_labels = np.argmax(logits_np, axis=1)

    print("Predicted labels shape:", predicted_labels.shape)
    print("First 10 predicted labels:", predicted_labels[:10])

    class_labels = checkpoint["class_labels"].detach().cpu().numpy()
    per_fault_accuracy = {}

    for fault_label in class_labels:
        total_for_fault = 0
        correct_for_fault = 0

        for i in range(total_examples):
            true_label = true_labels_np[i]
            predicted_label = predicted_labels[i]

            if true_label == fault_label:
                total_for_fault += 1

                if predicted_label == fault_label:
                    correct_for_fault += 1

        if total_for_fault == 0:
            fault_accuracy = 0.0
        else:
            fault_accuracy = correct_for_fault / total_for_fault

        per_fault_accuracy[int(fault_label)] = {
            "correct": correct_for_fault,
            "total": total_for_fault,
            "accuracy": fault_accuracy
        }

    num_fault_classes = len(class_labels)

    label_to_matrix_index = {}
    for index in range(num_fault_classes):
        fault_label = int(class_labels[index])
        label_to_matrix_index[fault_label] = index

    confusion_matrix = np.zeros((num_fault_classes, num_fault_classes), dtype=np.int32)

    for i in range(total_examples):
        true_label = int(true_labels_np[i])
        predicted_label = int(predicted_labels[i])

        true_row_index = label_to_matrix_index[true_label]
        predicted_col_index = label_to_matrix_index[predicted_label]

        confusion_matrix[true_row_index, predicted_col_index] += 1

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    print("\nOverall results:")
    print("Total evaluated examples:", total_examples)
    print("Number of classes:", num_classes)
    print("Top-1 accuracy:", round(topk_accuracies[1] * 100, 2), "%")
    print("Top-2 accuracy:", round(topk_accuracies[2] * 100, 2), "%")
    print("Top-3 accuracy:", round(topk_accuracies[3] * 100, 2), "%")

    print("\nPer-fault results:")
    print("Fault | Correct | Total | Accuracy")
    for fault_label in class_labels:
        fault_result = per_fault_accuracy[int(fault_label)]
        print(
            str(int(fault_label)).rjust(5), "|",
            str(fault_result["correct"]).rjust(7), "|",
            str(fault_result["total"]).rjust(5), "|",
            str(round(fault_result["accuracy"] * 100, 2)).rjust(8), "%"
        )

    print("\nConfusion matrix summary:")
    print("Rows = true labels")
    print("Cols = predicted labels")
    print("Class order:", class_labels.tolist())

    confusion_matrix_df = pd.DataFrame(
        confusion_matrix,
        index=class_labels.tolist(),
        columns=class_labels.tolist()
    )

    print("\nConfusion matrix table:")
    print(confusion_matrix_df.to_string())

    confusion_matrix_csv_path = os.path.join(
        processed_data_directory,
        "confusion_matrix.csv"
    )

    with open(confusion_matrix_csv_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)

        header_row = ["true_label \\ predicted_label"]
        for fault_label in class_labels:
            header_row.append(int(fault_label))
        csv_writer.writerow(header_row)

        for row_index in range(num_fault_classes):
            true_fault_label = int(class_labels[row_index])
            row_values = [true_fault_label]

            for col_index in range(num_fault_classes):
                row_values.append(int(confusion_matrix[row_index, col_index]))

            csv_writer.writerow(row_values)

    print("\nSaved confusion matrix CSV to:", confusion_matrix_csv_path)

    return


if __name__ == "__main__":
    main()
