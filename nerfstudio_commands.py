import subprocess
import sys
import time
import argparse


def run_command(command):
    print(f"Running command: {' '.join(command)}")
    process = subprocess.run(command)
    if process.returncode != 0:
        print(f"Command failed: {' '.join(command)}")
        sys.exit(process.returncode)

def invoke_command(input_path, output_path, colmap_model_path=None, skip_colmap=False, verbose=False):
    # Step 1: Process the data
    process_data_cmd = [
        "ns-process-data",
        "images",  # change to "images" if processing images
        "--data",
        input_path,
        "--output-dir",
        output_path,
    ]
    
    if skip_colmap:
        process_data_cmd.append("--skip-colmap")
    if colmap_model_path:
        process_data_cmd.append("--colmap-model-path")
        process_data_cmd.append(colmap_model_path)
    if verbose:
        process_data_cmd.append("--verbose")
    
    run_command(process_data_cmd)

    print("Data processing complete.")
    time.sleep(2)  # Optionally wait a bit

    # Step 2: Train using splatfacto
    train_cmd = ["ns-train", "splatfacto", "--data", output_path]
    run_command(train_cmd)

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Nerfstudio commands.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the raw data.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for processed data.")
    parser.add_argument("--colmap-model-path", type=str, help="Path to the COLMAP model directory.")
    parser.add_argument("--skip-colmap", action="store_true", help="Skip COLMAP processing.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()
    invoke_command(args.data_path, args.output_dir, args.colmap_model_path, args.skip_colmap, args.verbose)