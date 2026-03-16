"""
Simple evaluation script outline.

This file is intentionally pseudocode only.
It shows the evaluation flow in one place using inline comments.
"""

from pathlib import Path
import csv
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from models.simplemlp import SimpleMLP



def main():
    # 1) Parse command line arguments
    #    Example args:
    #    --checkpoint-path path/to/model.pt
    #    --data-path path/to/eval_data.npz
    #    --split full or validation_runs
    #    --batch-size 64
    #    --confusion-csv-path path/to/confusion_matrix.csv
    # training data csv paths
    current_file_path = Path(__file__).resolve()
    current_directory = current_file_path.parent
    parent_directory = current_directory.parent
    parent_directory = os.path.abspath(parent_directory)
    data_directory = os.path.join(parent_directory, "data")
    raw_data_directory = os.path.join(data_directory, "raw")
    processed_data_directory = os.path.join(data_directory, "processed")

    model_file_path = os.path.join(parent_directory, "baseline_mlp.pt")
    data_file_path = os.path.join(processed_data_directory, "small_fault0_20_runs1_500_W60_step10.npz")


    # 2) Pick device for inference
    #    if MPS is available, use it; otherwise use CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # 3) Load checkpoint (.pt)
    #    Required checkpoint keys:
    #    - model_state_dict
    #    - input_dim
    #    - num_classes
    #    - class_labels (torch tensor)
    #    - normalization_mean (torch tensor)
    #    - normalization_std (torch tensor)
    #    - validation_runs (torch tensor)
    #    Also read metadata if needed:
    #    - data_file_name, window_size, step_size, epochs
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
        "input_dim",
        "num_classes",
        "class_labels",
        "normalization_mean",
        "normalization_std",
        "validation_runs",
    ]
    missing_keys = [key for key in required_checkpoint_keys if key not in checkpoint]
    if len(missing_keys) > 0:
        raise KeyError("Checkpoint is missing required keys: " + str(missing_keys))

    print("Loaded checkpoint:", model_file_path)
    print("Checkpoint keys:", sorted(checkpoint.keys()))
    

    # 4) Rebuild model and load weights
    input_dim = int(checkpoint["input_dim"])
    num_classes = int(checkpoint["num_classes"])

    model = SimpleMLP(input_dim=input_dim, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print("Model loaded.")
    print("Model input_dim:", input_dim)
    print("Model num_classes:", num_classes)

    # 5) Load evaluation dataset (.npz)
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

    # 6) Select rows to evaluate
    # change this to "full" if you want all rows
    split_mode = "validation_runs"

    if split_mode == "full":
        inputs_selected = inputs
        answers_selected = answers
        run_ids_selected = run_ids
    elif split_mode == "validation_runs":
        # checkpoint["validation_runs"] is a torch tensor.
        # detach() -> remove gradient tracking (safe for inference values).
        # cpu()    -> move tensor data to CPU memory.
        # numpy()  -> convert tensor to a NumPy array so np.isin can use it.
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

    # 7) Normalize evaluation inputs using checkpoint stats
    #    mean = checkpoint["normalization_mean"].cpu().numpy()
    #    std = checkpoint["normalization_std"].cpu().numpy()
    #    inputs_norm = (inputs_selected - mean) / std
    normalization_mean = checkpoint["normalization_mean"].detach().cpu().numpy()
    normalization_std = checkpoint["normalization_std"].detach().cpu().numpy()

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

    inputs_norm = (inputs_selected - normalization_mean) / normalization_std

    print("Normalization applied using checkpoint stats.")
    print("Normalized input min/max:", float(np.min(inputs_norm)), float(np.max(inputs_norm)))

    # 8) Build DataLoader for evaluation (shuffle=False)
    #    Convert inputs_norm and answers_selected to tensors
    #    Create TensorDataset and DataLoader
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

    # 9) Run inference and collect outputs
    #    with torch.no_grad():
    #        for each batch:
    #            logits = model(batch_inputs)
    #            store logits and true labels
    #    Combine all batch results into full arrays
    # We do "inference" here (not training):
    # - forward pass only
    # - no gradient tracking
    # - collect model outputs for ALL examples
    #
    # Why collect logits?
    # - top-k accuracy needs ranked class scores (not just one argmax label)
    # - confusion matrix/per-fault accuracy need true + predicted labels
    #   (predictions will be derived from logits in later steps)
    all_logits = []
    all_true_labels = []

    # torch.no_grad() makes evaluation faster and uses less memory.
    with torch.no_grad():
        for batch_inputs, batch_answers in evaluation_loader:
            # Move this batch to same device as model (MPS or CPU).
            batch_inputs = batch_inputs.to(device)
            batch_answers = batch_answers.to(device)

            # Forward pass: model returns raw class scores (logits).
            batch_logits = model(batch_inputs)

            # Move results back to CPU so we can combine everything later
            # and convert to NumPy for metric calculations.
            all_logits.append(batch_logits.cpu())
            all_true_labels.append(batch_answers.cpu())

    if len(all_logits) == 0:
        raise ValueError("No batches found in evaluation loader.")

    # Merge batch pieces into one full tensor per output type.
    logits_tensor = torch.cat(all_logits, dim=0)
    true_labels_tensor = torch.cat(all_true_labels, dim=0)

    # Convert to NumPy arrays for simple metric code in later steps.
    logits_np = logits_tensor.numpy()
    true_labels_np = true_labels_tensor.numpy()

    print("Inference done.")
    print("logits shape:", logits_np.shape)
    print("true labels shape:", true_labels_np.shape)

    # 10) Compute top-k accuracies
    #     top-1 accuracy
    #     top-2 accuracy
    #     top-3 accuracy
    #     (use k <= num_classes)
    num_classes = logits_np.shape[1]
    total_examples = logits_np.shape[0]

    # logits_np is like an array of rows
    # each row is the output of the model
    # ex. [2.1, -0.7, 0.3, 4.8, ..., 1.2]
    # which every value is the largest is the predicted class

    # true_labels_np is the actual class for each row of predictions

    # k must be less than or equal to number of classes
    topk_accuracies = {}
    k_values = [1, 2, 3]

    for k in k_values:
        # Sort class scores from highest to lowest for each example.
        sorted_class_indices = np.argsort(logits_np, axis=1) # indices of ascending order
        sorted_class_indices = sorted_class_indices[:, ::-1] # reverse the order to now be descending

        # Keep only the best k class indices for each example.
        topk_class_indices = sorted_class_indices[:, :k]
        # this keeps all the rows
        # but only the first k columns of each row

        # count the number of correct predictions
        # it is correct if it is 
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

    # 11) Compute top-1 predicted labels
    #     predicted = argmax(logits, axis=1)
    # Top-1 prediction means:
    # for each example, choose the single class with the highest score.
    predicted_labels = np.argmax(logits_np, axis=1)

    print("Predicted labels shape:", predicted_labels.shape)
    print("First 10 predicted labels:", predicted_labels[:10])

    # 12) Compute per-fault accuracy
    #     class_labels = checkpoint["class_labels"].cpu().numpy().tolist()
    #     for each fault label in class_labels:
    #         total = count(true == label)
    #         correct = count((true == label) and (predicted == label))
    #         accuracy = correct / total
    class_labels = checkpoint["class_labels"].detach().cpu().numpy()

    # Store accuracy information for each fault number.
    # Example:
    # per_fault_accuracy[5] = {"correct": 80, "total": 100, "accuracy": 0.80}
    per_fault_accuracy = {}

    for fault_label in class_labels:
        total_for_fault = 0
        correct_for_fault = 0

        for i in range(total_examples):
            true_label = true_labels_np[i]
            predicted_label = predicted_labels[i]

            # Only count examples that truly belong to this fault.
            if true_label == fault_label:
                total_for_fault += 1

                # Count it as correct only if prediction matches too.
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

    print("\nPer-fault accuracy:")
    for fault_label in class_labels:
        fault_result = per_fault_accuracy[int(fault_label)]
        print(
            "Fault", int(fault_label),
            ":", fault_result["correct"], "correct",
            ":", fault_result["total"], "total",
            ":", round(fault_result["accuracy"] * 100, 2), "%"
        )
    

    # 13) Build confusion matrix
    #     rows = true labels
    #     cols = predicted labels
    #     class order = checkpoint["class_labels"]
    num_fault_classes = len(class_labels)

    # Map each fault label to its row/column index in the matrix.
    # Example: if class_labels = [0, 1, 2], then:
    # label_to_matrix_index[0] = 0
    # label_to_matrix_index[1] = 1
    # label_to_matrix_index[2] = 2
    label_to_matrix_index = {}
    for index in range(num_fault_classes):
        fault_label = int(class_labels[index])
        label_to_matrix_index[fault_label] = index

    # Matrix starts with all zeros.
    # rows = true class
    # cols = predicted class
    confusion_matrix = np.zeros((num_fault_classes, num_fault_classes), dtype=np.int32)

    for i in range(total_examples):
        true_label = int(true_labels_np[i])
        predicted_label = int(predicted_labels[i])

        true_row_index = label_to_matrix_index[true_label]
        predicted_col_index = label_to_matrix_index[predicted_label]

        confusion_matrix[true_row_index, predicted_col_index] += 1

    print("\nConfusion matrix shape:", confusion_matrix.shape)
    print("Class order used for rows/columns:", class_labels.tolist())
    print("Confusion matrix:")
    print(confusion_matrix)


    # 14) Print results
    #     print number of examples
    #     print top-1/top-2/top-3
    #     print per-fault accuracy table
    #     print confusion matrix summary
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

    # Use a DataFrame so row/column labels line up like a real matrix.
    confusion_matrix_df = pd.DataFrame(
        confusion_matrix,
        index=class_labels.tolist(),
        columns=class_labels.tolist()
    )

    print("\nConfusion matrix table:")
    print(confusion_matrix_df.to_string())

    # 15) Save confusion matrix to CSV
    #     first row should be header with predicted labels
    #     first column should be true labels
    #     write counts as integers
    confusion_matrix_csv_path = os.path.join(
        processed_data_directory,
        "confusion_matrix.csv"
    )

    with open(confusion_matrix_csv_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)

        # Header row:
        # first cell explains the matrix layout,
        # remaining cells are predicted class labels.
        header_row = ["true_label \\ predicted_label"]
        for fault_label in class_labels:
            header_row.append(int(fault_label))
        csv_writer.writerow(header_row)

        # Each row starts with the true class label,
        # followed by the counts for each predicted class.
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
