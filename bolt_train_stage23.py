import os
import sys
import subprocess
import tarfile
import zipfile
import time
import shutil
from datetime import datetime
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, **kwargs):
        return iterable

import concurrent.futures

def process_single_dataset(name, s3_dir_url, target_dir):
    """Processes a single dataset: download, extract, and cleanup."""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Started processing dataset: {name}")
    
    temp_dl_dir = os.path.join(target_dir, f"temp_{name}")
    os.makedirs(temp_dl_dir, exist_ok=True)
    
    # 1. Download
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Downloading parts for {name} from {s3_dir_url}...")
    cmd = [
        "conductor", "s3", "cp", s3_dir_url, temp_dl_dir, 
        "--recursive", 
        "--exclude", "*", 
        "--include", f"{name}_processed.tar.part_*",
        "--only-show-errors"
    ]
    start_time = time.time()
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to download {name}: {e}")
        return False
        
    dl_elapsed = time.time() - start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Download {name} finished in {dl_elapsed:.1f} seconds.")
    
    # 2. Extract
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Extracting {name}_processed ...")
    parts = sorted([os.path.join(temp_dl_dir, f) for f in os.listdir(temp_dl_dir) if f.startswith(f"{name}_processed.tar.part_")])
    
    if not parts:
        print(f"WARNING: No parts found for {name}!")
        return False
        
    cat_cmd = ["cat"] + parts
    tar_cmd = ["tar", "xf", "-", "-C", target_dir]
    
    ext_start_time = time.time()
    try:
        cat_process = subprocess.Popen(cat_cmd, stdout=subprocess.PIPE)
        tar_process = subprocess.Popen(tar_cmd, stdin=cat_process.stdout)
        cat_process.stdout.close()  # Allow cat_process to receive a SIGPIPE if tar_process exits
        tar_process.communicate()
        
        if tar_process.returncode != 0:
            raise subprocess.CalledProcessError(tar_process.returncode, tar_cmd)
    except Exception as e:
        print(f"ERROR: Failed to extract {name}: {e}")
        return False
        
    ext_elapsed = time.time() - ext_start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Extraction {name} finished in {ext_elapsed:.1f} seconds.")
        
    # 3. Cleanup
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Cleaning up {len(parts)} downloaded parts for {name}...")
    for part in parts:
        os.remove(part)
    os.rmdir(temp_dl_dir)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished {name}.\n")
    return True

def fetch_and_extract_multipart_datasets(s3_dir_url, target_dir):
    """Downloads multi-part dataset archives from S3 and extracts them efficiently using parallel processing."""
    os.makedirs(target_dir, exist_ok=True)
    
    datasets = ["arkitscenes", "blendedmvs", "co3d", "habitat", "megadepth", "scannetpp", "staticthings3d", "waymo", "wildrgbd"]
    
    max_workers = min(len(datasets), 4) # Adjust max_workers as needed, 4 is usually a safe default for I/O
    print(f"Starting parallel download and extraction with {max_workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_dataset, name, s3_dir_url, target_dir): name for name in datasets}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(datasets), desc="Total Datasets Progress"):
            name = futures[future]
            try:
                success = future.result()
                if not success:
                    print(f"Failed to fully process {name}")
            except Exception as exc:
                print(f'{name} generated an exception: {exc}')

def run_stage(stage_num, cmd, env=None):
    print(f"==========================================")
    print(f"          Starting Stage {stage_num}         ")
    print(f"==========================================")
    print("Running training command:")
    print(" ".join(cmd))
    
    # Use existing environment, update with any passed env variables
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
        
    subprocess.check_call(cmd, env=run_env)
    print(f"==========================================")
    print(f"          Finished Stage {stage_num}         ")
    print(f"==========================================\n")

def get_torchrun_prefix():
    """Constructs the torchrun prefix string handling both single-node and multi-node setup."""
    nproc_per_node = os.environ.get("NPROC_PER_NODE", "8")
    nnodes = os.environ.get("NNODES", "1")
    node_rank = os.environ.get("NODE_RANK", "0")
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")

    prefix = ["torchrun", f"--nproc_per_node={nproc_per_node}"]
    
    # If running on multiple nodes, append multi-node args
    if int(nnodes) > 1:
        prefix.extend([
            f"--nnodes={nnodes}",
            f"--node_rank={node_rank}",
            f"--master_addr={master_addr}",
            f"--master_port={master_port}"
        ])
        
    prefix.append("train.py")
    return prefix

def main():
    print("Setting up environment...")
    
    local_cache_dir = os.environ.get('COREFLOW_DATA_ROOT', '/mnt/train_data')
    data_dir = os.path.join(local_cache_dir, "dust3r_processed_data")

    s3_dataset_url = os.environ.get("DUST3R_S3_DATASET", "s3://qianfan-shen/dust3r/dust3r_processed_data/")
    
    # 2. Data fetching and uncompress
    if not os.path.exists(data_dir):
        print(f"Fetching data from {s3_dataset_url}...")
        fetch_and_extract_multipart_datasets(s3_dataset_url, data_dir)
    else:
        print(f"Data already exists at {data_dir}, skipping download.")

    
    # 3. Download pretrained weights and Stage 1 checkpoint
    print("Downloading Stage 1 checkpoint from S3...")
    os.makedirs("checkpoints", exist_ok=True)
    
    stage1_ckpt_path = "checkpoints/checkpoint-best-stage1.pth"
    s3_ckpt_url = os.environ.get("DUST3R_S3_CKPT", "s3://qianfan-shen/dust3r/checkpoints/stage1_224/checkpoint-best.pth") # Update this URL in your env vars if needed

    if not os.path.exists(stage1_ckpt_path):
        try:
            print(f"Fetching Stage 1 checkpoint from {s3_ckpt_url}...")
            subprocess.check_call([
                "conductor", "s3", "cp", s3_ckpt_url, stage1_ckpt_path
            ])
            print("Successfully downloaded Stage 1 checkpoint.")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to download checkpoint from {s3_ckpt_url}: {e}")
            print("Please ensure the S3 URL is correct and the file exists.")
            sys.exit(1)
    else:
        print(f"Stage 1 checkpoint already exists at {stage1_ckpt_path}.")


    # 4. Set base output directory for checkpoints and logs
    base_output_dir = os.environ.get('BOLT_ARTIFACT_DIR', 'checkpoints/dust3r_train_output')
    print(f"Base saving directory for checkpoints and logs: {base_output_dir}")

    # Launch TensorBoard in the background
    tensorboard_port = os.environ.get('TENSORBOARD_PORT', '6006')
    
    # Only launch TensorBoard on the master node to prevent port conflicts on multi-node runs
    node_rank = os.environ.get("NODE_RANK", "0")
    if node_rank == "0":
        print(f"Starting TensorBoard in the background on port {tensorboard_port}, logging to {base_output_dir}...")
        subprocess.Popen([
            sys.executable, "-m", "tensorboard.main",
            f"--logdir={base_output_dir}", 
            f"--port={tensorboard_port}", 
            "--bind_all"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    torchrun_prefix = get_torchrun_prefix()
    
    total_gpus = int(os.environ.get("NNODES", "1")) * int(os.environ.get("NPROC_PER_NODE", "8"))
    
    # Dynamic accumulation to stabilize global batch size
    # Stage 1 original: 16 (bs) * 1 (accum) * 8 (gpus) = 128
    # Stage 2/3 original: 4 (bs) * 2 (accum) * 8 (gpus) = 64
    stage1_accum = 1
    stage23_accum = max(1, 16 // total_gpus) if total_gpus > 8 else 2 # E.g. 8 GPUs -> 2, 32 GPUs -> 1

    print("Skipping Stage 1 (already trained).")

    ###################################################################
    # Stage 2: 512 linear
    ###################################################################
    stage2_output = os.path.join(base_output_dir, "stage2_512")
    if node_rank == "0":
        os.makedirs(stage2_output, exist_ok=True)
    
    train_dataset_str_512 = (
        f" + 10_000 @ Habitat(1_000_000, ROOT='{data_dir}/habitat_processed', split='train', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)"
        f" + 10_000 @ BlendedMVS(ROOT='{data_dir}/blendedmvs_processed', split='train', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)"
        f" + 10_000 @ MegaDepth(ROOT='{data_dir}/megadepth_processed', split='train', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)"
        f" + 10_000 @ ARKitScenes(ROOT='{data_dir}/arkitscenes_processed', split='train', aug_crop=256, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)"
        f" + 10_000 @ Co3d(ROOT='{data_dir}/co3d_processed', split='train', aug_crop=16, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)"
        f" + 10_000 @ StaticThings3D(ROOT='{data_dir}/staticthings3d_processed', aug_crop=256, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)"
        f" + 10_000 @ ScanNetpp(ROOT='{data_dir}/scannetpp_processed', split='train', aug_crop=256, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) "
        f" + 10_000 @ Waymo(ROOT='{data_dir}/waymo_processed', split='train', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) "
        f" + 10_000 @ WildRGBD(ROOT='{data_dir}/wildrgbd_processed', split='train', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) "
    )

    test_dataset_str_512 = (
        f" 1_000 @ Habitat(10_000, ROOT='{data_dir}/habitat_processed', split='val', resolution=(512,384), seed=777)"
        f" + 1_000 @ BlendedMVS(ROOT='{data_dir}/blendedmvs_processed', split='val', resolution=(512,384), seed=777)"
        f" + 1_000 @ MegaDepth(ROOT='{data_dir}/megadepth_processed', split='val', resolution=(512,336), seed=777)"
        f" + 1_000 @ Co3d(ROOT='{data_dir}/co3d_processed', split='test', resolution=(512,384), seed=777) "
    )

    stage1_best_ckpt = "checkpoints/checkpoint-best-stage1.pth"

    cmd_stage2 = torchrun_prefix + [
        f"--train_dataset={train_dataset_str_512}",
        f"--test_dataset={test_dataset_str_512}",
        "--train_criterion=ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)",
        "--test_criterion=Regr3D_ScaleShiftInv(L21, gt_scale=True)",
        "--model=AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)",
        f"--pretrained={stage1_best_ckpt}",
        "--lr=0.0001",
        "--min_lr=1e-06",
        "--warmup_epochs=20",
        "--epochs=100",
        "--batch_size=4",
        f"--accum_iter={stage23_accum}",
        "--save_freq=10",
        "--keep_freq=10",
        "--eval_freq=1",
        "--print_freq=10",
        f"--output_dir={stage2_output}"
    ]

    run_stage(2, cmd_stage2)

    ###################################################################
    # Stage 3: 512 dpt
    ###################################################################
    stage3_output = os.path.join(base_output_dir, "stage3_512dpt")
    if node_rank == "0":
        os.makedirs(stage3_output, exist_ok=True)
    
    stage2_best_ckpt = os.path.join(stage2_output, "checkpoint-best.pth")

    cmd_stage3 = torchrun_prefix + [
        f"--train_dataset={train_dataset_str_512}",
        f"--test_dataset={test_dataset_str_512}",
        "--train_criterion=ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)",
        "--test_criterion=Regr3D_ScaleShiftInv(L21, gt_scale=True)",
        "--model=AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)",
        f"--pretrained={stage2_best_ckpt}",
        "--lr=0.0001",
        "--min_lr=1e-06",
        "--warmup_epochs=15",
        "--epochs=90",
        "--batch_size=4",
        f"--accum_iter={stage23_accum}",
        "--save_freq=5",
        "--keep_freq=10",
        "--eval_freq=1",
        "--print_freq=10",
        "--disable_cudnn_benchmark",
        f"--output_dir={stage3_output}"
    ]

    run_stage(3, cmd_stage3)

    print("All stages completed successfully!")

if __name__ == "__main__":
    main()