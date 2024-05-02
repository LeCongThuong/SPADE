import os
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm


def inference_model(epoch, result_dir):
    command = [
        "python3", "test.py",
        "--name", "spade",
        "--dataset_mode", "custom",
        "--image_dir", "/mnt/hmi/thuong/wb_train_val_test_dataset/valid/np_depth_512",
        "--which_epoch", str(epoch),
        "--results_dir", result_dir,
        "--label_dir", "/mnt/hmi/thuong/wb_train_val_test_dataset/valid/print_512"
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise Exception(f"An error occurred: {result.stderr}")
    

def convert_np_to_ply(np_dir, ply_dir):
    command = [
        "python3", "convert_np_to_ply.py",
        "-np", np_dir,
        "-ply", ply_dir
    ]

    # Execute the command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check for errors
    if result.returncode != 0:
        raise Exception(f"Error occurred: {result.stderr}")
    
def run_evaluation(gt_dir, pred_dir, result_file):
    command = [
        "python3", "evaluation.py",
        "-gt", gt_dir,
        "-pred", pred_dir,
        "-res", result_file
    ]

    # Execute the command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check for errors
    if result.returncode != 0:
        raise Exception(f"Error occurred: {result.stderr}")

def main():
    epochs = list(range(1, 11))

    for epoch in tqdm(epochs):
        epoch = str(epoch)
        try:
            result_dir = f"/mnt/hmi/thuong/SPADE/results/spade_validation/epoch_{epoch}"
            Path(result_dir).mkdir(exist_ok=True, parents=True)
            inference_model(epoch=epoch, result_dir=result_dir)

            np_dir = f"/mnt/hmi/thuong/SPADE/results/spade_validation/epoch_{epoch}/np"
            ply_dir = f"/mnt/hmi/thuong/SPADE/results/spade_validation/epoch_{epoch}/ply_512"
            Path(ply_dir).mkdir(exist_ok=True, parents=True)
            convert_np_to_ply(np_dir, ply_dir)

            gt_dir = "/mnt/hmi/thuong/wb_train_val_test_dataset/valid/"
            pred_dir = f"/mnt/hmi/thuong/SPADE/results/spade_validation/epoch_{epoch}"
            result_file = f"/mnt/hmi/thuong/SPADE/results/spade_validation/epoch_{epoch}/metrics_score.json"
            run_evaluation(gt_dir, pred_dir, result_file)
            print(f"Done: Epoch {epoch}")

        except Exception as e:
            print(f"Error at {epoch}: {e}")


if __name__ == "__main__":
    main()