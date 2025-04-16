import os
import argparse
import subprocess
import sys
import time
import shutil
import time

RETRY_LIMIT = 3
RETRY_COUNTER = 0
RETRY_COOLDOWN = 180  # seconds

def run_command(command):
    print(f"Running command: {' '.join(command)}")
    process = subprocess.run(command)
    if process.returncode != 0:
        print(f"Command failed: {' '.join(command)}")
        sys.exit(process.returncode)


def full_pipe(video_path, frame_output_dir, frame_count, skip_colmap=False,
              max_num_iterations=30000, start_over=False, only_nerfstudio=False,
              nerfstudio_model="splatfacto", advanced_training=False, use_mcmc=False, num_downscales=8):
    
    frame_extract_start_time = time.time()
    print("Starting full pipeline...")
    print(f"Video path: {video_path}") #data/data_source/camera.MP4
    print(f"Output path: {frame_output_dir}") #outputs/full_pipe/camera/images
    print(f"Frame count: {frame_count}")
    print(f"Use only nerfstudio: {only_nerfstudio}")
    print(f"Nerftsudio Model: {nerfstudio_model}")
    # Check if the output directory exists, if not create it
    if not os.path.exists(frame_output_dir):
        os.makedirs(frame_output_dir)
        
    skip_frame_extraction = False
    skip_mast3r_processing = False
    frame_extract_time = 0
    mast3r_processing_time = 0
    training_time = 0
    
    #Check for existing file
    if os.path.exists(frame_output_dir):
        #Check for images in the folder
        if len(os.listdir(frame_output_dir)) != 0:
            print(f"Output directory {frame_output_dir} already exists and is not empty.")
            if start_over:
                #Clean all content
                print(f"Deleting all content in {frame_output_dir} as start_over is True.")
                for filename in os.listdir(frame_output_dir):
                    file_path = os.path.join(frame_output_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')
            else:
                print(f"Frames already present in {frame_output_dir} skipping frame extraction")
                skip_frame_extraction = True
                
        else:
            print(f"Output directory {frame_output_dir} exists but is empty. Continuing.")
    
    # Step 1: Extract frames from video
    # frame_extract_cmd = [
    #     "sfextract",
    #     video_path,  
    #     "--frame-count",
    #     frame_count,
    #     "--output",
    #     frame_output_dir,
    # ]
    
    frame_extract_cmd = [
        "python",
        "nerfstudio_commands.py",
        "--data-path",
        video_path,
        "--output-dir",
        frame_output_dir,
        "--frame-count",
        str(frame_count),
        "--frame-extraction",
    ]
    if skip_frame_extraction is False and only_nerfstudio is False:
        run_command(frame_extract_cmd)
        frame_extract_time = time.time() - frame_extract_start_time
        time.sleep(60)


    # Step 2: Process the data with Mast3r
    mast3r_output_dir = frame_output_dir.split("/images")[0] # outputs/full_pipe/camera
    print(f"Output directory for Mast3r: {mast3r_output_dir}")
    
    #Check if exist colmap/sparse/0 and its content made up of three files
    colmap_dir = os.path.join(mast3r_output_dir, "colmap", "sparse", "0")
    if os.path.exists(colmap_dir):
        #Check if there are three files with dimensions > 1KB
        colmap_files = os.listdir(colmap_dir)
        colmap_files = [f for f in colmap_files if os.path.isfile(os.path.join(colmap_dir, f)) and os.path.getsize(os.path.join(colmap_dir, f)) > 1024]
        if len(colmap_files) == 3:
            print(f"Colmap results directory {colmap_dir} already exists and contains valid files.")
            if start_over:
                #Delete colmap/sparse/0
                print(f"Deleting colmap results directory {colmap_dir} as start_over is True.")
                shutil.rmtree(colmap_dir)
            else:
                print(f"Colmap results directory {colmap_dir} already exists and contains valid files. Skipping Mast3r processing.")
                skip_mast3r_processing = True
    
    mast3r_glomap_command = [
        "conda", "run", "--live-stream", "-n", "mast3r_env",
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
        str(20),
        "--win_cyclic",
    ]
    if skip_mast3r_processing is False and only_nerfstudio is False:
        mast3r_start_time = time.time()
        run_command(mast3r_glomap_command)
        mast3r_processing_time = time.time() - mast3r_start_time
        time.sleep(60)
        
    print("Data processing complete.")
    
    #Check if transform.json is already present in the directory
    transform_json_path = os.path.join(mast3r_output_dir, "transform.json")
    if os.path.exists(transform_json_path):
        print(f"Transform.json already exists in {mast3r_output_dir}. Proceeding to delete old files")
        #Delete transform.json
        os.remove(transform_json_path)
        os.remove(os.path.join(mast3r_output_dir, "sparse_pc.ply"))
        #Delete images, images_2, images_4, and images_8 folders
        for folder in ["images", "images_2", "images_4", "images_8", "export", "models"]:
            folder_path = os.path.join(mast3r_output_dir, folder)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
    else:
        print(f"Transform.json does not exist in {mast3r_output_dir}. Proceeding with nerfstudio processing.")
    
    # Step 3: Process the data and train with nerfstudio
    if only_nerfstudio:
        print("Only nerfstudio processing is selected. Skipping Mast3r processing.")
        nerfstudio_cmd = [
            "python",
            "nerfstudio_commands.py",
            "--data-path",
            video_path,
            "--output-dir",
            f"{mast3r_output_dir}",            
            "--max-num-iterations",
            str(max_num_iterations),
            "--model",
            f"{nerfstudio_model}",
        ]
    else:
        nerfstudio_cmd = [
            "python",
            "nerfstudio_commands.py",
            "--data-path",
            frame_output_dir,
            "--output-dir",
            f"{mast3r_output_dir}",
            "--colmap-model-path",
            "colmap/sparse/0",
            "--skip-colmap",
            "--max-num-iterations",
            str(max_num_iterations),
            "--model",
            f"{nerfstudio_model}",
        ]
       
    nerfstudio_cmd.append("--num-downscales")
    nerfstudio_cmd.append(str(num_downscales))
     
    if advanced_training:
        nerfstudio_cmd.append("--advanced")
        
    if use_mcmc:
        nerfstudio_cmd.append("--use-mcmc")
    
    training_start_time = time.time()
    run_command(nerfstudio_cmd)
    training_time = time.time() - training_start_time
    print("Nerfstudio processing complete.")
    
    if skip_frame_extraction is False:
        print(f"Frame extraction time: {time.strftime('%H:%M:%S', time.gmtime(frame_extract_time))}")
        
    if skip_mast3r_processing is False:
        print(f"Mast3r processing time: {time.strftime('%H:%M:%S', time.gmtime(mast3r_processing_time))}")
        
    print(f"Nerfstudio training time: {time.strftime('%H:%M:%S', time.gmtime(training_time))}")
    print("Full pipeline complete.")
    print("Total time: ", time.strftime("%H:%M:%S", time.gmtime(frame_extract_time + mast3r_processing_time + training_time)))

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run Complete Gaussian Splatting pipeline.")
#     parser.add_argument(
#         "--video-path", type=str, required=True, help="Path to the raw data."
#     )
#     parser.add_argument(
#         "--output-dir", type=str, required=True, help="Directory for processed data."
#     )
#     parser.add_argument(
#         "--frame-count", type=str, help="Path to the COLMAP model directory."
#     )
#     parser.add_argument(
#         "--skip-colmap", action="store_true", help="Skip COLMAP processing."
#     )
#     parser.add_argument(
#         "--max-num-iterations", type=int, default=30000, help="Maximum number of iterations for training."
#     )
#     parser.add_argument(
#         "--nerfstudio-model", type=str, default="splatfacto", choices=["splatfacto", "splatfacto-big", "splatfacto-w-light"], help="Model type to use for training."
#     )
#     parser.add_argument(
#         "--start-over", type=bool, default=False, help="Start over the pipeline."
#     )
#     parser.add_argument(
#         "--only-nerfstudio", type=bool, default=False, help="Use only nerfstudio for the entire pipeline"
#     )
#     parser.add_argument(
#         "--advanced-training", action="store_true", help="Enable advanced settings for training."
#     )
#     parser.add_argument(
#         "--use-mcmc", action="store_true", help="Use MCMC for training."
#     )
#     parser.add_argument(
#         "--num-downscales",
#         type=int,
#         default=8,
#         choices=[1, 2, 4, 8],
#         help="Number of downscales for processing.",
#     )
#     args = parser.parse_args()
    
        
#     for attempt in range(1, RETRY_LIMIT + 1):
#         try:
#             full_pipe(
#                 video_path=args.video_path,
#                 frame_output_dir=args.output_dir,
#                 frame_count=args.frame_count,
#                 skip_colmap=args.skip_colmap,
#                 max_num_iterations=args.max_num_iterations,
#                 start_over=args.start_over,
#                 only_nerfstudio=args.only_nerfstudio,
#                 nerfstudio_model=args.nerfstudio_model,
#                 advanced_training=args.advanced_training,
#                 use_mcmc=args.use_mcmc,
#                 num_downscales=args.num_downscales,
#             )
#             print("Pipeline completed successfully.")
#             break  # Exit the loop if successful
#         except Exception as e:
#             print(f"Attempt {attempt} failed: {e}")
#             if attempt <= RETRY_LIMIT:
#                 print(f"Retrying in {RETRY_COOLDOWN} seconds...")
#                 time.sleep(RETRY_COOLDOWN)
#             else:
#                 print("Max attempts reached. Exiting.")
#                 raise e

