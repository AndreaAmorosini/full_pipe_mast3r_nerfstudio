import subprocess
import sys
import time
import argparse
import re
import os

config_file_path = ""

def run_command(command):
    print(f"Running command: {' '.join(command)}")
    process = subprocess.run(command)
    # with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1) as process:
    #     for line in process.stdout:
    #         print(line, end="")
    #         if "Training Finished" in line:
    #             print("Training finished.")
    #     process.wait()
    if process.returncode != 0:
        print(f"Command failed: {' '.join(command)}")
        sys.exit(process.returncode)

def invoke_command(input_path, output_path, colmap_model_path=None, skip_colmap=False, max_num_iterations = 30000, verbose=False):
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
    model_output_path = f"{output_path}/export"
    train_cmd = ["ns-train", "splatfacto", "--data", output_path, "--output-dir", model_output_path, "--viewer.quit-on-train-completion", "True"]
    if max_num_iterations != 30000:
        train_cmd.append("--max-num-iterations")
        train_cmd.append(str(max_num_iterations))
    run_command(train_cmd)
    
    config_file_path = f"{model_output_path}"
    config_file_path_1 = os.listdir(config_file_path)[0]
    config_file_path_2 = os.listdir(config_file_path)[0]
    config_file_path_3 = os.listdir(config_file_path)[0]
    final_config_file_path = os.path.join(config_file_path, config_file_path_1, config_file_path_2, config_file_path_3, "config.yml")
    
    #Step 3: export final .ply
    export_cmd = [
        "ns-export",
        "gaussian-splat",
        "--load-config",
        f"{model_output_path}/config.yaml",
        "--output-dir",
        f"{output_path}/model",
    ]
    run_command(export_cmd)
    print(".ply exported to", f"{output_path}/exports/splat/")
    
    print("Pipe complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Nerfstudio commands.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the raw data.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for processed data.")
    parser.add_argument("--colmap-model-path", type=str, help="Path to the COLMAP model directory.")
    parser.add_argument("--skip-colmap", action="store_true", help="Skip COLMAP processing.")
    parser.add_argument("--max-num-iterations", type=int, default=30000, help="Maximum number of iterations for training.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()
    invoke_command(args.data_path, args.output_dir, args.colmap_model_path, args.skip_colmap, args.max_num_iterations, args.verbose)