import os
import extract_frames
import nerfstudio_commands
import mast3r_glomap_cli
import argparse
import subprocess
import sys
import time

def run_command(command):
    print(f"Running command: {' '.join(command)}")
    process = subprocess.run(command)
    if process.returncode != 0:
        print(f"Command failed: {' '.join(command)}")
        sys.exit(process.returncode)


def full_pipe(video_path, frame_output_dir, frame_count, skip_colmap, max_num_iterations=30000):
    
    # Check if the output directory exists, if not create it
    if not os.path.exists(frame_output_dir):
        os.makedirs(frame_output_dir)
    
    # Step 1: Extract frames from video
    extract_frames.invoke_command(video_path, frame_output_dir, frame_count)

    # Step 2: Process the data with Mast3r
    mast3r_output_dir = frame_output_dir.split("/input")[0]
    print(f"Output directory for Mast3r: {mast3r_output_dir}")
    mast3r_glomap_command = [
        "python",
        "mast3r_glomap_cli.py",
        "--model_name",
        "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        "--input_files",
        frame_output_dir,
        "--output_dir",
        mast3r_output_dir,
        "--scenegraph_type",
        "swin",
        "--winsize",
        15,
        "--win_cyclic",
    ]
    run_command(mast3r_glomap_command)
    print("Data processing complete.")
    time.sleep(2)  # Optionally wait a bit
    
    # Step 3: Process the data and train with nerfstudio
    nerfstudio_commands.invoke_command(frame_output_dir, mast3r_output_dir, colmap_model_path="colmap/sparse/0", skip_colmap=True, max_num_iterations=max_num_iterations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Nerfstudio commands.")
    parser.add_argument(
        "--video-path", type=str, required=True, help="Path to the raw data."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory for processed data."
    )
    parser.add_argument(
        "--frame-count", type=str, help="Path to the COLMAP model directory."
    )
    parser.add_argument(
        "--skip-colmap", action="store_true", help="Skip COLMAP processing."
    )
    parser.add_argument(
        "--max-num-iterations", type=int, default=30000, help="Maximum number of iterations for training."
    )
    args = parser.parse_args()
    
    full_pipe(
        video_path=args.video_path,
        frame_output_dir=args.output_dir,
        frame_count=args.frame_count,
        skip_colmap=args.skip_colmap,
        max_num_iterations=args.max_num_iterations,
    )
