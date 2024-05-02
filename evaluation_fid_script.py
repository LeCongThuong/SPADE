import subprocess


def calculate_fid(depth_path):
# Define the command as a list of arguments
    command = [
        'python3', '-m', 'pytorch_fid',
        depth_path,
        '/mnt/hmi/thuong/wb_train_val_test_dataset/valid/depth_512'
    ]

    # Execute the command and raise an error if it fails
    try:
        subprocess.run(command, check=True)
        print("Command executed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        raise

def main():
    for epoch in list(range(1, 11)):
        try:
            print(f"Start epoch: {str(epoch)}")
            depth_dir = f'/mnt/hmi/thuong/SPADE/results/spade_validation/epoch_{str(epoch)}/depth_512'
            calculate_fid(depth_dir)
            print("Done!")
        except Exception as e:
            print(f"Error occurred at epoch {str(epoch)}: {e}")

if __name__ == "__main__":
    main()
