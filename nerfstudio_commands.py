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

def invoke_command(input_path, output_path, colmap_model_path=None, skip_colmap=False, max_num_iterations = 30000, verbose=False, model="splatfacto", advanced=False):	
    # Step 1: Process the data
    process_data_cmd = [
        "ns-process-data",
        "images" if skip_colmap is True else "video",
        "--gpu", 
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
    train_cmd = ["ns-train", model, "--data", output_path, "--output-dir", model_output_path, "--viewer.quit-on-train-completion", "True"]
    if max_num_iterations != 30000:
        train_cmd.append("--max-num-iterations")
        train_cmd.append(str(max_num_iterations))

    if advanced:
        train_cmd.extend([
            "--pipeline.model.cull-alpha-thresh", "0.005",
            "--pipeline.model.use-scale-regularization", "True",
            "--pipeline.model.use-bilateral-grid", "True",
            "--mixed-precision", "True",
            "--pipeline.datamanager.train-cameras-sampling-strategy", "fps",
            "--pipeline.model.camera-optimizer.mode", "SO3xR3",
            "--pipeline.model.color-corrected-metrics", "True",
        ])
    # --pipeline.model.cull-alpha-thresh FLOAT :threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality. (default: 0.1)
    # --pipeline.model.use-scale-regularization {True,False}: If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians. (default: False)
    # --pipeline.model.use-bilateral-grid {True,False}: If True, use bilateral grid to handle the ISP changes in the image space. This technique was introduced in the paper 'Bilateral Guided Radiance Field Processing' (https://bilarfpro.github.io/). (default:False)
    # --pipeline.model.strategy {default,mcmc}: The default strategy will be used if strategy is not specified. Other strategies, e.g. mcmc, can be used. (default: default)
    # --pipeline.model.noise-lr FLOAT: MCMC samping noise learning rate. Default to 5e5. (default:500000.0)
    # --pipeline.model.mcmc-opacity-reg FLOAT: Regularization term for opacity in MCMC strategy.(default: 0.01)
    # --pipeline.model.mcmc-scale-reg FLOAT: Regularization term for scale in MCMC strategy.(default: 0.01)
    # --mixed-precision {True,False}: Whether or not to use mixed precision for training. (default: False)
    # --pipeline.datamanager.cache-images {cpu,gpu,disk}: Where to cache images in memory. (default: gpu)
    # --pipeline.datamanager.train-cameras-sampling-strategy {random,fps}: Specifies which sampling strategy is used to generate train cameras,'random' means sampling uniformly random without replacement, 'fps' means farthest point sampling which is helpful to reduce the artifact due to oversampling subsets of cameras that are very close to each other. (default: random)
    # --pipeline.model.camera-optimizer.mode {off,SO3xR3,SE3}: Pose optimization strategy to use. If enabled, we recommend SO3xR3. (default: off)
    # --pipeline.model.num-downscales INT: at the beginning, resolution is 1/2^d, where d is this number (default: 2)
    # --pipeline.model.output-depth-during-training {True,False}: If True, output depth during training. Otherwise, only output depth during evaluation. (default: False)
    # --pipeline.model.color-corrected-metrics {True,False}: If True, apply color correction to the rendered images before computing the metrics. (default: False)
        

    run_command(train_cmd)
    
    config_file_path = f"{model_output_path}"
    config_file_path_1 = os.listdir(config_file_path)[0]
    config_file_path_2 = os.listdir(config_file_path + "/" + config_file_path_1)[0]
    config_file_path_3 = os.listdir(config_file_path + "/" + config_file_path_1 + "/" + config_file_path_2)[0]
    final_config_file_path = os.path.join(config_file_path, config_file_path_1, config_file_path_2, config_file_path_3, "config.yml")
    
    #Step 3: export final .ply
    export_cmd = [
        "ns-export",
        "gaussian-splat",
        "--load-config",
        f"{final_config_file_path}",
        "--output-dir",
        f"{output_path}/splat",
    ]
    run_command(export_cmd)
    print(".ply exported to", f"{output_path}/splat")
    
    print("Pipe complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Nerfstudio commands.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the raw data.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for processed data.")
    parser.add_argument("--colmap-model-path", type=str, help="Path to the COLMAP model directory.")
    parser.add_argument("--skip-colmap", action="store_true", help="Skip COLMAP processing.")
    parser.add_argument("--num-downscales", type=int, default=8, help="Number of downscales for processing.")
    parser.add_argument("--max-num-iterations", type=int, default=30000, help="Maximum number of iterations for training.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--model", type=str, default="splatfacto", choices=["splatfacto", "splatfacto-big", "splatfacto-w", "splatfacto-w-light"], help="Model type to use for training.")
    parser.add_argument("--advanced", action="store_true", help="Enable advanced settings for training.")
    args = parser.parse_args()
    invoke_command(args.data_path, args.output_dir, args.colmap_model_path, args.skip_colmap, args.max_num_iterations, args.verbose, args.model, args.advanced)