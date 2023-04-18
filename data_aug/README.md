# Data Augmentation with Style Transfer
This is a Python script that performs data augmentation on a folder of images using various techniques, including style transfer. It requires Python 3 and the following Python packages:

- numpy
- torch
- torchvision
- Pillow
- opencv-python
- argparse
- tqdm

## Usage
To use the script, run the following command in a terminal or command prompt:
```
python data_aug.py source_folder target_folder augmentation
```
Replace source_folder with the path to the folder containing the images you want to augment, target_folder with the path to the folder where you want to save the augmented images, and augmentation with the name of the data augmentation technique you want to apply to the images.

The available data augmentation techniques are:

- rotate
- flip_horizontal
- flip_vertical
- cutout
- blur
- brightness
- contrast
- color
- sharpness
- autocontrast
- equalize
- invert
- style_transfer
- style_transfer_mix
## Example
To perform style transfer data augmentation on a folder of images in source_images and save the augmented images to a folder target_images, run the following command:
```
python data_aug.py source_images target_images style_transfer
```
The script will apply style transfer to each image in source_images and save the augmented images with the prefix style_transfer_ to target_images.

## License
This code is licensed under the MIT License. Feel free to use, modify, and distribute it as you wish.
