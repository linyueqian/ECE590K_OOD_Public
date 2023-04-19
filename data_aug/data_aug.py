
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 23:17:22 2023

@author: Yueqian Lin
"""

import os

import numpy as np
import torch
import torchvision.transforms as tvs_trans
from PIL import Image as Image
from PIL import ImageEnhance, ImageOps
import cv2
import argparse
from styleaug import StyleAugmentor
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm
import pixmix_utils as pixmix_utils

# PyTorch Tensor <-> PIL Image transforms:
toTensor = ToTensor()
toPIL = ToPILImage()

def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Define the available data augmentation techniques
AVAILABLE_AUGMENTATIONS = {
    'rotate': lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
    'flip_horizontal': lambda img: cv2.flip(img, 1),
    'flip_vertical': lambda img: cv2.flip(img, 0),
    'cutout': lambda img: cutout(img, n_holes=1, length=5),
    'blur': lambda img: cv2.GaussianBlur(img, (5, 5), 0),
    'brightness': lambda img: pil_to_cv2(ImageEnhance.Brightness(Image.fromarray(img)).enhance(1.5)),
    'contrast': lambda img: pil_to_cv2(ImageEnhance.Contrast(Image.fromarray(img)).enhance(1.5)),
    'color': lambda img: pil_to_cv2(ImageEnhance.Color(Image.fromarray(img)).enhance(1.5)),
    'sharpness': lambda img: pil_to_cv2(ImageEnhance.Sharpness(Image.fromarray(img)).enhance(1.5)),
    'autocontrast': lambda img: pil_to_cv2(ImageOps.autocontrast(Image.fromarray(img))),
    'equalize': lambda img: pil_to_cv2(ImageOps.equalize(Image.fromarray(img))),
    'invert': lambda img: pil_to_cv2(ImageOps.invert(Image.fromarray(img))),
    'style_transfer': lambda img: style_transfer(img),
    'style_transfer_mix': lambda img: style_transfer_mix(img),
}

def style_transfer(img):
    # Get original image size and resize to 256x256
    size = img.size
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    
    # permute
    img = cv2_to_pil(img)

    # Convert image to tensor and move to GPU
    img_tensor = toTensor(img).unsqueeze(0)
    img_tensor = img_tensor.to('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create style augmentor
    augmentor = StyleAugmentor()

    # Randomize style
    img_restyled = augmentor(img_tensor)

    # Downsample to original size
    img_restyled = torch.nn.functional.interpolate(img_restyled, size=(32, 32), mode='bicubic', align_corners=False)

    # Convert tensor back to cv2 image and return
    img_restyled = img_restyled.squeeze().cpu()
    img_restyled = pil_to_cv2(toPIL(img_restyled))
    
    return img_restyled

def style_transfer_mix(img, beta=0.8):
    origin_tensor = toTensor(img).unsqueeze(0).to('cuda:0' if torch.cuda.is_available() else 'cpu')
     # Get original image size and resize to 256x256
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    
    # permute
    img = cv2_to_pil(img)

    # Convert image to tensor and move to GPU
    img_tensor = toTensor(img).unsqueeze(0).to('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create style augmentor
    augmentor = StyleAugmentor()

    # Randomize style
    img_restyled = augmentor(img_tensor)
    # Downsample to 32*32
    img_restyled = torch.nn.functional.interpolate(img_restyled, size=(32, 32), mode='bicubic', align_corners=False)
    mixings = pixmix_utils.mixings
    mixed_op = np.random.choice(mixings)
    mixed = mixed_op(origin_tensor, img_restyled, beta)
    mixed = torch.clip(mixed, 0, 1)
    
    
    # Convert tensor back to cv2 image and return
    mixed = mixed.squeeze().cpu()
    mixed = pil_to_cv2(toPIL(mixed))
    # save image
    img_restyled = img_restyled.squeeze().cpu()
    img_restyled = pil_to_cv2(toPIL(img_restyled))
    cv2.imwrite('./aug_img/style_transfer_mix_before_0029.png', img_restyled)
    return mixed

    

# Define the available data augmentation techniques
def cutout(img, n_holes=1, length=40):
    """
    Randomly mask out one or more patches from an image.

    Args:
        img (ndarray): The input image.
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.

    Returns:
        The augmented image.
    """
    h, w = img.shape[:2]

    mask = np.ones((h, w), np.float32)

    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.

    mask = np.stack([mask] * 3, axis=2)
    img = img * mask

    return img

def augment_images(source_folder, target_folder, augmentation):
    '''
    Return a function that performs the chosen augmentation technique on all images in the source folder
    
    Parameters:
        source_folder (str): Path to the folder containing the images to be augmented
        target_folder (str): Path to the folder where the augmented images will be saved
        augmentation (str): The name of the augmentation technique to be applied
    '''
    # Create target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Iterate over all image files in source folder
    for filename in tqdm(os.listdir(source_folder)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load the image
            image_path = os.path.join(source_folder, filename)
            image = cv2.imread(image_path)

            # Apply the chosen augmentation technique
            if augmentation in AVAILABLE_AUGMENTATIONS:
                augmented_image = AVAILABLE_AUGMENTATIONS[augmentation](image)
            else:
                print(f'Error: Unknown augmentation {augmentation}')
                return

            # Save the augmented image
            target_path = os.path.join(target_folder, f'{augmentation}_{filename}')
            cv2.imwrite(target_path, augmented_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform basic data augmentation on a folder of images.')
    parser.add_argument('source_folder', type=str, help='Path to the source folder containing the images.')
    parser.add_argument('target_folder', type=str, help='Path to the target folder to save the augmented images.')
    parser.add_argument('augmentation', type=str, choices=AVAILABLE_AUGMENTATIONS.keys(),
                        help='The data augmentation technique to apply to the images.')

    args = parser.parse_args()

    augment_images(args.source_folder, args.target_folder, args.augmentation)