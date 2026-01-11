import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.utils as vutils

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from models import FFA


# ---------------- METRICS ----------------
def compute_metrics(pred, gt):
    """
    pred, gt: torch tensors [3, H, W] in range [0,1]
    """
    pred_np = pred.permute(1, 2, 0).numpy()
    gt_np = gt.permute(1, 2, 0).numpy()

    psnr_val = psnr(gt_np, pred_np, data_range=1.0)
    ssim_val = ssim(gt_np, pred_np, data_range=1.0, channel_axis=2)

    return psnr_val, ssim_val


# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='its', help='its or ots')
    parser.add_argument('--test_imgs', type=str, required=True,
                        help='Path to Smokebench/Test folder')
    args = parser.parse_args()

    BASE_DIR = os.getcwd()
    TEST_DIR = args.test_imgs

    HAZY_DIR = os.path.join(TEST_DIR, 'hazy')
    CLEAR_DIR = os.path.join(TEST_DIR, 'clear')

    assert os.path.exists(HAZY_DIR), f"Missing folder: {HAZY_DIR}"
    assert os.path.exists(CLEAR_DIR), f"Missing folder: {CLEAR_DIR}"

    # ---------------- OUTPUT ----------------
    output_dir = os.path.join(BASE_DIR, f'pred_FFA_{args.task}')
    os.makedirs(output_dir, exist_ok=True)

    print("Pred dir :", output_dir)
    print("Hazy dir :", HAZY_DIR)
    print("Clear dir:", CLEAR_DIR)

    # ---------------- MODEL ----------------
    gps = 3
    blocks = 19
    model_path = os.path.join(
        BASE_DIR, f'trained_models/{args.task}_train_ffa_{gps}_{blocks}.pk'
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    net = FFA(gps=gps, blocks=blocks)
    net = nn.DataParallel(net)
    net.load_state_dict(checkpoint['model'])
    net.eval()
    net.to(device)

    # ---------------- TRANSFORMS ----------------
    input_transform = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58],
                      std=[0.14, 0.15, 0.152])
    ])

    to_tensor = tfs.ToTensor()

    # ---------------- TEST LOOP ----------------
    psnr_vals, ssim_vals = [], []

    files = sorted(os.listdir(HAZY_DIR))

    for idx, fname in enumerate(files, 1):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        hazy_path = os.path.join(HAZY_DIR, fname)
        clear_path = os.path.join(CLEAR_DIR, fname)

        if not os.path.exists(clear_path):
            print(f"\n⚠️  Missing GT: {fname}, skipping")
            continue

        haze = Image.open(hazy_path).convert('RGB')
        clear = Image.open(clear_path).convert('RGB')

        haze_tensor = input_transform(haze).unsqueeze(0).to(device)
        clear_tensor = to_tensor(clear)

        with torch.no_grad():
            pred = net(haze_tensor)

        pred = torch.clamp(pred.squeeze(0).cpu(), 0, 1)

        # -------- METRICS --------
        p, s = compute_metrics(pred, clear_tensor)
        psnr_vals.append(p)
        ssim_vals.append(s)

        # -------- SAVE --------
        save_path = os.path.join(output_dir, fname)
        vutils.save_image(pred, save_path)

        print(f"[{idx}] {fname} | PSNR: {p:.2f} dB | SSIM: {s:.4f}")

    # ---------------- RESULTS ----------------
    print("\n========== FINAL RESULTS ==========")
    print(f"Average PSNR : {np.mean(psnr_vals):.2f} dB")
    print(f"Average SSIM : {np.mean(ssim_vals):.4f}")
    print("===================================")


if __name__ == "__main__":
    main()
