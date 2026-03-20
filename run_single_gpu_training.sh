#!/bin/bash
# Single A100 Training Script for DUSt3R (Step 1: 224 resolution)

# Ensure we are in the correct directory
cd /root/workspace/recgen/dust3r || exit 1

# 1. Download the starting pretrained CroCo weights
echo "Checking for CroCo pretrained weights..."
mkdir -p checkpoints/
wget -nc https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTLarge_BaseDecoder.pth -P checkpoints/

# 2. Run the single GPU training process
echo "Starting DUSt3R training on 1 GPU..."
torchrun --nproc_per_node=1 train.py \
    --train_dataset=" + 100_000 @ Habitat(1_000_000, ROOT='/mnt/qianfan_data/dust3r_processed_data/habitat_processed', split='train', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ BlendedMVS(ROOT='/mnt/qianfan_data/dust3r_processed_data/blendedmvs_processed', split='train', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ MegaDepth(ROOT='/mnt/qianfan_data/dust3r_processed_data/megadepth_processed', split='train', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ ARKitScenes(ROOT='/mnt/qianfan_data/dust3r_processed_data/arkitscenes_processed', split='train', aug_crop=256, resolution=224, transform=ColorJitter) + 100_000 @ Co3d(ROOT='/mnt/qianfan_data/dust3r_processed_data/co3d_processed', split='train', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ StaticThings3D(ROOT='/mnt/qianfan_data/dust3r_processed_data/staticthings3d_processed', aug_crop=256, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ ScanNetpp(ROOT='/mnt/qianfan_data/dust3r_processed_data/scannetpp_processed', split='train', aug_crop=256, resolution=224, transform=ColorJitter) " \
    --test_dataset=" 1_000 @ Habitat(10_000, ROOT='/mnt/qianfan_data/dust3r_processed_data/habitat_processed', split='val', resolution=224, seed=777) + 1_000 @ BlendedMVS(ROOT='/mnt/qianfan_data/dust3r_processed_data/blendedmvs_processed', split='val', resolution=224, seed=777) + 1_000 @ MegaDepth(ROOT='/mnt/qianfan_data/dust3r_processed_data/megadepth_processed', split='val', resolution=224, seed=777) + 1_000 @ Co3d(ROOT='/mnt/qianfan_data/dust3r_processed_data/co3d_processed', split='test', mask_bg='rand', resolution=224, seed=777) " \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth" \
    --lr=0.0001 \
    --min_lr=1e-06 \
    --warmup_epochs=10 \
    --epochs=100 \
    --batch_size=32 \
    --accum_iter=4 \
    --save_freq=5 \
    --keep_freq=10 \
    --eval_freq=1 \
    --output_dir="checkpoints/dust3r_224_single_gpu"


