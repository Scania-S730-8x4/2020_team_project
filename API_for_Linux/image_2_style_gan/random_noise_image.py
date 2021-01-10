import argparse
import torch
import os
from torchvision.utils import save_image


def random_pixel_image(trg_dir='images/aligned/', x=1024, y=1024, min_float=0.0, max_float=1.0):
    random_tensor = torch.rand((1, 3, x, y))
    random_tensor = torch.clamp(random_tensor, min_float, max_float)

    return random_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Target_dir, x(Dimension X), y(Dimension Y), min(float), max(float)')
    parser.add_argument('--trg_dir', default='images/aligned/')
    parser.add_argument('--x', default=1024, type=int)
    parser.add_argument('--y', default=1024, type=int)
    parser.add_argument('--min', default=0.0, type=float)
    parser.add_argument('--max', default=1.0, type=float)
    args = parser.parse_args()

    trg_dir = args.trg_dir
    min_float = args.min
    max_float = args.max
    x = args.x
    y = args.y

    if not os.path.isdir(trg_dir):
            os.makedirs(trg_dir, exist_ok=True)

    random_tensor = random_pixel_image(trg_dir, x, y, min_float, max_float)
    n = len(os.listdir(trg_dir)) + 1
    save_image(random_tensor, trg_dir + f'random_noise_{n}.png')