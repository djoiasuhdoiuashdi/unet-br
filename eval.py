import argparse
import os

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.nn.parallel import DataParallel
from torchvision import transforms as T
from tqdm import tqdm
from model.model import UNetBR
from utils.util import crop  # Make sure this is used or remove if unnecessary


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--input_dir', type=str,default="./dataset/input", help='Directory containing images to remove background')
    parser.add_argument('--output_dir', type=str, default="./output", help='Directory to save processed images')
    parser.add_argument('--load', help='Path to load weights from', type=str, default='./weight/weights.pth')
    parser.add_argument('--num_block', help='Number of UNet blocks', type=int, default=2)
    parser.add_argument('--extensions', nargs='+', default=['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'],
                        help='List of image file extensions to process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    load = args.load
    num_block = args.num_block
    extensions = args.extensions

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Gather all image files in the input directory with specified extensions
    files = []
    for ext in extensions:
        files.extend([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(ext)])

    if not files:
        print(f"No image files found in {input_dir} with extensions {extensions}.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_transforms = T.Compose([
        T.RandomInvert(p=1.0),  # Ensure probability parameter is specified
        T.Grayscale(num_output_channels=1),  # Specify number of output channels
        T.ToTensor(),
    ])

    model = UNetBR(num_block)
    model = DataParallel(model)
    checkpoint = torch.load(load, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for fname in tqdm(files, desc="Processing images", total=len(files)):
            try:
                # Load and preprocess the image
                img = Image.open(fname).convert('RGB')  # Ensure image is in RGB
                img = data_transforms(img)
                img = img.unsqueeze(0).to(device)

                # Forward pass
                output = model(img)

                # Assuming output is a list and you want the last element
                out = output[-1].squeeze(0)  # Remove batch dimension

                # Post-process the output (e.g., invert the transform if needed)
                processed_img = T.ToPILImage()(out.cpu())

                gray_np = np.array(processed_img)
                inverted = cv2.bitwise_not(gray_np)
                _, otsu = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Prepare output filename
                base_name = os.path.basename(fname)
                output_path = os.path.join(output_dir, base_name)

                # Save the processed image
                cv2.imwrite(output_path, otsu)
            except Exception as e:
                print(f"Failed to process {fname}: {e}")


if __name__ == "__main__":
    main()