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
    if process.returncode != 0:
        print(f"Command failed: {' '.join(command)}")
        sys.exit(process.returncode)


def invoke_command(
    video_path,
    output_path,
    frame_count=5,
):
    print(f"Video path: {video_path}")
    print(f"Output path: {output_path}")
    print(f"Frame count: {frame_count}")
    # Step 1: Process the data
    process_data_cmd = [
        "sfextract",
        video_path,  
        "--frame-count",
        frame_count,
        "--output",
        output_path,
    ]

    run_command(process_data_cmd)

    print("Frame extraction complete.")
    time.sleep(2)  # Optionally wait a bit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Frame Extraction.")
    parser.add_argument(
        "--video-path", type=str, required=True, help="Path to the raw data."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory for processed data."
    )
    parser.add_argument(
        "--frame-count", type=int, help="Number of frames to extract and use"
    )
    args = parser.parse_args()
    invoke_command(
        args.video_path,
        args.output_dir,
        args.frame_count,
    )
