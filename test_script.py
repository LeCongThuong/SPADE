import os
import shutil
import subprocess
import argparse
from pathlib import Path


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
    
def parse_aug():
    parser = argparse.ArgumentParser(prog='Convert numpy depth to ply 3D object')
    parser.add_argument('-pred', '--pred_dir', help='path to numpy file')
    parser.add_argument('-res', '--res_path', help='path to save result file')
    args = parser.parse_args()
    return args


def main():
    gt_dir = "/mnt/hmi/thuong/wb_train_val_test_dataset/test/"
    args = parse_aug()
    run_evaluation(gt_dir, args.pred_dir, args.res_path)


if __name__ == "__main__":
    main()