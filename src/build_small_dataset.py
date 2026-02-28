# this script will take the TEP dataset from the raw CSV
# then turn it into windows and save that as a compressed .npz file
# this dataset will be what the model trains on

import os
from pathlib import Path
import time
import numpy as np
import pandas as pd

# Column names of TEP data
FAULT_COL = "faultNumber" # what to predict
RUN_COL = "simulationRun" # which simulation run
SAMPLE_COL = "sample" # the time step / sample within the run

# Sensor columns below
# sensors xmeas_1 to xmeas_41
XMEAS_COLS = []
for i in range(1, 42):
    XMEAS_COLS.append(f"xmeas_{i}")

# sensors xmv_1 to xmv_11
XMV_COLS = []
for i in range(1, 12):
    XMV_COLS.append(f"xmv_{i}")

SENSOR_COLS = XMEAS_COLS + XMV_COLS

# final columns list of tep data csv
TEP_COLS = [FAULT_COL, RUN_COL, SAMPLE_COL] + SENSOR_COLS

def load_one_run(csv_path, fault_number, run_id, chunk_size=500000):
    """
    Reads only ONE simulationRun from a big CSV.

    In this dataset, simulationRun repeats for each faultNumber,
    so the unique identifier is:
        (faultNumber, simulationRun)

    chunk_size is how many rows to read at once
    because the CSV is millions of rows

    Parameters:
      csv_path: path to CSV
      fault_number: integer 0..20
      run_id: integer 1..500
      chunk_size: how many rows to read at once
    
    Returns:
        df: dataframe containing all rows of sensor data where
            only that faultNumber and simulationRun match
            sorted by sample number (time)

    """

    # check if files exist
    if not os.path.exists(csv_path):
        raise FileNotFoundError("File not found: " + csv_path)
    
    data_pieces = []
    reader = pd.read_csv(
        csv_path,
        usecols=TEP_COLS,
        chunksize=chunk_size
    )

    for chunk in reader:
        # rows where faultNumber and simulationRun match
        filtered = chunk[
            (chunk[FAULT_COL] == fault_number)
            & (chunk[RUN_COL] == run_id)
        ]

        # if any rows found then store them in pieces
        if len(filtered) > 0:
            data_pieces.append(filtered)
    
    if len(data_pieces) == 0:
        raise ValueError(
            f"No rows found for faultNumber = {fault_number} and simulationRun = {run_id}"
        )
    
    # combine all the data pieces together for one df
    df = pd.concat(data_pieces, ignore_index=True)
    df = df.sort_values(SAMPLE_COL) # sort by sample (time taken)
    df = df.reset_index(drop=True) # new index numbers, drop previous ones

    return df


# turn df generated from one run into training examples
# each run_df is of the same fault_number
def make_windows(run_df, faultNum, runNum, window_size, step_size):
    """
    Split ONE run (run_df) into many overlapping windows.

    A "window" is a chunk of consecutive rows
    Each window becomes ONE training example
    Each window has rows of the same fault

    Returns:
      windows: list of windows
        each window = list of rows
        each row = list of sensor values
      fault_nums: list of ints (one fault per window)
    """

    windows = [] # list of windows
    faults = [] # the fault number (answer) corresponding to that window training data

    print(f"For faultNumber {faultNum}, simulationRun {runNum}, Length of run_df is {len(run_df)}")

    samples_count = len(run_df) # should be 500 samples per run
    start_of_window = 0
    while start_of_window + window_size <= samples_count:
        end_of_window = start_of_window + window_size
        window_df = run_df.iloc[start_of_window:end_of_window]
        window_df = window_df.sort_values(SAMPLE_COL).reset_index(drop=True)
        window_sensor_df = window_df[SENSOR_COLS]
        window_as_list = window_sensor_df.values.tolist()
        window = window_as_list

        windows.append(window)
        faults.append(faultNum)

        start_of_window += step_size

    return windows, faults

def convert_windows_to_vectors(windows):
    vectors = []
    for window in windows:
        vector = []
        for sample in window:
            vector.extend(sample)
        vectors.append(vector)
    return vectors


def main():
    # data used will be from small set of specific faultNumbers and simulationRuns
    faults_to_use = list(range(0, 6)) #0..5, out of total 0-20 faults
    runs_to_use = list(range(1, 21)) # 1..20, total there are 500 runs per fault in training

    # Window settings
    window_size = 60
    step_size = 10

    # training data csv paths
    current_file_path = Path(__file__).resolve()
    current_directory = current_file_path.parent
    parent_directory = current_directory.parent
    parent_directory = os.path.abspath(parent_directory)
    data_directory = os.path.join(parent_directory, "data")
    raw_data_directory = os.path.join(data_directory, "raw")
    processed_data_directory = os.path.join(data_directory, "processed")

    faultfree_training_csv_path = os.path.join(
        raw_data_directory,
        "TEP_FaultFree_Training.csv"
    )
    faulty_training_csv_path = os.path.join(
        raw_data_directory,
        "TEP_Faulty_Training.csv"
    )
    output_directory = processed_data_directory
    os.makedirs(output_directory, exist_ok=True)

    first_fault = faults_to_use[0]
    last_fault = faults_to_use[-1]
    first_run = runs_to_use[0]
    last_run = runs_to_use[-1]

    output_filename = f"small_fault{first_fault}_{last_fault}_runs{first_run}_{last_run}_W{window_size}_step{step_size}.npz"
    output_path = os.path.join(output_directory, output_filename)


    # print stats
    print("Building small dataset...")
    print("Faults:", faults_to_use)
    print("Runs:", runs_to_use)
    print("Window size:", window_size)
    print("Step size:", step_size)
    print("Saving to:", output_path)

    # timer
    start_time = time.time()

    # to store windows from all runs
    all_inputs = []         # list of flattened window vectors (each length 3120)
    all_answers = []         # list of answers / fault numbers (0..20)
    all_run_ids = []        # list of simulationRun IDs (same length as all_answers)  

    # loop through the faults
    for fault in faults_to_use:
        # choose csv path, fault free or faulty 
        if fault == 0:
            csv_path = faultfree_training_csv_path
        else:
            csv_path = faulty_training_csv_path

        # loop through the runs
        for simRun in runs_to_use:
            print("\nLoading run...")
            print(f"faultNumber = {fault}, simulationRun = {simRun}")

            # load the simulationRun with that faultNumber
            run_df = load_one_run(
                csv_path=csv_path,
                fault_number=fault,
                run_id=simRun
            )

            
            windows, faults = make_windows(
                run_df=run_df,
                faultNum=fault,
                runNum=simRun,
                window_size=window_size,
                step_size=step_size
            )

            # currently this gives windows
            # windows is a list of lists
            # each window contains 500 samples in order
            inputs = convert_windows_to_vectors(windows)
            answers = faults
            run_ids = [simRun] * len(answers)

            all_inputs.extend(inputs)
            all_answers.extend(answers)
            all_run_ids.extend(run_ids)

        continue

    # now we have all the jits
    # convert to numpy arrays
    inputs_np_array = np.array(all_inputs, dtype=np.float32)
    answers_np_array = np.array(all_answers, dtype=np.int8)
    run_ids_np_array = np.array(all_run_ids, dtype=np.int16)

    faults_used_np_array = np.array(faults_to_use, dtype=np.int8)
    runs_used_np_array = np.array(runs_to_use, dtype=np.int16)

    source_files_np_array = np.array(
        [
            faultfree_training_csv_path,
            faulty_training_csv_path
        ],
        dtype=np.str_
    )
    

    print("\nDone building.")
    print("Total number of input vectors:", len(inputs_np_array))
    print("Inputs shape:", inputs_np_array.shape)
    print("Answers shape:", answers_np_array.shape)
    print("Run IDs shape:", run_ids_np_array.shape)

    # save everything
    np.savez_compressed(
        file=output_path,

        inputs=inputs_np_array,
        answers=answers_np_array,
        run_ids=run_ids_np_array,

        window_size=window_size,
        step_size=step_size,
        
        faults_used=faults_used_np_array,
        runs_used=runs_used_np_array,

        source_files=source_files_np_array,
    )

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\nSaved dataset to:", output_path)
    print(f"Time elapsed: {minutes} min {seconds} sec")

    return


if __name__ == "__main__":
    main()

