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
    'brightness': lambda img: ImageEnhance.Brightness(Image.fromarray(img)).enhance(1.5),
    'contrast': lambda img: ImageEnhance.Contrast(Image.fromarray(img)).enhance(1.5),
    'color': lambda img: ImageEnhance.Color(Image.fromarray(img)).enhance(1.5),
    'sharpness': lambda img: ImageEnhance.Sharpness(Image.fromarray(img)).enhance(1.5),
    'autocontrast': lambda img: ImageOps.autocontrast(Image.fromarray(img)),
    'equalize': lambda img: ImageOps.equalize(Image.fromarray(img)),
    'invert': lambda img: ImageOps.invert(Image.fromarray(img)),
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
    img_restyled = torch.nn.functional.interpolate(img_restyled, size=size, mode='bicubic', align_corners=False)

    # Convert tensor back to cv2 image and return
    img_restyled = img_restyled.squeeze().cpu()
    img_restyled = pil_to_cv2(toPIL(img_restyled))
    
    return img_restyled

def style_transfer_mix(img, beta=0.2):
     # Get original image size and resize to 256x256
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    
    # permute
    img = cv2_to_pil(img)

    # Convert image to tensor and move to GPU
    img_tensor = toTensor(img).unsqueeze(0)
    size = img.size
    img_tensor = img_tensor.to('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create style augmentor
    augmentor = StyleAugmentor()

    # Randomize style
    img_restyled = augmentor(img_tensor)
    # Downsample to original size
    img_restyled = torch.nn.functional.interpolate(img_restyled, size=size, mode='bicubic', align_corners=False)
    mixings = pixmix_utils.mixings
    mixed_op = np.random.choice(mixings)
    mixed = mixed_op(img_tensor, img_restyled, beta=0.5)
    mixed = torch.clip(mixed, 0, 1)
    
    
    # Convert tensor back to cv2 image and return
    mixed = mixed.squeeze().cpu()
    mixed = pil_to_cv2(toPIL(mixed))
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