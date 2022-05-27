import os
import os.path as osp
import random
import shutil

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def load_watermark(img_original, watermark_path, watermark_files, prob=1):
    """
    Loads a random watermark on original image and worded image

    Parameters
    -----------
    img_original : torch.Tensor
        raw image from dataset
    watermark_path : str
        Path containing watermarks
    watermark_files : list
        Contains watermark filenames
    prob: probability of loading watermark

    Returns
    -------
    img_watermark : torch.Tensor
        Image loaded with watermarks, for generating binary masks
    """
    # some images will not have watermarks loaded
    if random.random() > prob:
        return img_original.clone()

    _, img_width, img_height = img_original.shape
    img_watermark = img_original.clone()

    logo_id = random.randint(0, len(watermark_files) - 1)
    logo = Image.open(osp.join(watermark_path, watermark_files[logo_id]))
    logo = logo.convert('RGBA')

    # random rotation
    rotate_angle = random.randint(0, 360)
    logo_rotate = logo.rotate(rotate_angle, expand=True)

    # random logo size
    logo_height = int(img_height * (random.uniform(0.2, 0.5)))
    logo_width  = int(img_width  * (random.uniform(0.2, 0.5)))

    logo_resize = logo_rotate.resize((logo_height, logo_width))

    logo = transforms.ToTensor()(logo_resize)

    alpha = random.uniform(0.3, 0.6)
    start_width = random.randrange(0, img_width - logo_width)
    start_height = random.randrange(0, img_height - logo_height)

    img_watermark[:, start_width:(start_width+logo_width), start_height:(start_height+logo_height)] *= \
        (1.0 - alpha * logo[3:4, :, :]) + logo[:3, :, :] * alpha * logo[3:4, :, :]

    return img_watermark


def solve_mask(img, img_target):
    """
    Generates binary mask based from difference between img and img target
    Parameters
    ----------
    img : torch.Tensor
    img_target : torch.Tensor

    Returns
    -------
    mask : numpy.ndarray
    """
    img1 = np.asarray(img.permute(1, 2, 0).cpu())
    img2 = np.asarray(img_target.permute(1, 2, 0).cpu())
    img3 = abs(img1 - img2)
    mask = img3.sum(2) > (15.0 / 255.0)
    return mask.astype(int)


def main():
    dataset_dir = '../dataset'  # folder to store generated dataset, needs to contain photos and watermarks

    for label in ['train', 'val']:
        print(f">> Generating images for {label} set")

        results_path = f"{dataset_dir}/{label}"

        # required
        raw_path = osp.join(results_path, "raw")               # photos are unique between train/val/test
        watermark_path = osp.join(results_path, "watermarks")  # watermarks are unique between train/val/test

        # inputs
        raw_files = sorted([f for f in os.listdir(raw_path) if f[:-15] != 'Zone.Identifier'])
        watermark_files = sorted([f for f in os.listdir(watermark_path) if f[-15:] != 'Zone.Identifier'])

        # output paths
        watermark_mask_path = osp.join(results_path, 'watermark_mask')
        img_input_path = osp.join(results_path, 'watermark_input')

        shutil.rmtree(watermark_mask_path, ignore_errors=True)
        shutil.rmtree(img_input_path, ignore_errors=True)

        os.makedirs(watermark_mask_path, exist_ok=True)
        os.makedirs(img_input_path, exist_ok=True)

        i = 1
        blurrer = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(1,1))

        for photo in tqdm(raw_files):

            indiv_photo_path = osp.join(raw_path, photo)
            img = Image.open(indiv_photo_path)
            rgbimg = img.convert('RGB')
            img = rgbimg
            # img = img.resize((256, 256))

            # choose some of input images as hard negatives
            if random.random() < (1/2):
                img = np.array(img).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = ((torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                            .permute(2, 0, 1) / 255)
                            .to(torch.float32))
                img = blurrer(img)
                save_id = f'{i}.jpg'
                cv2.imwrite(osp.join(img_input_path, save_id),
                            cv2.cvtColor(np.array(img.permute(1, 2, 0) * 255), cv2.COLOR_BGR2RGB))
                i += 1

            else: # choose some of input images as positives
                img_original = transforms.ToTensor()(img)
                watermarked_img = load_watermark(img_original, watermark_path, watermark_files, prob=1)

                img_original = blurrer(img_original)
                watermarked_img = blurrer(watermarked_img)

                # # half of positive input images are stitched positives
                if random.random() < 0.5:
                    img_comb_free = torch.cat([img_original, img_original], dim=2)

                    if random.random() > 0.5:
                        # append img_original to the left
                        watermarked_comb_mask = torch.cat([watermarked_img, img_original], dim=2)
                    else:
                        # append img_original to the right
                        watermarked_comb_mask = torch.cat([img_original, watermarked_img], dim=2)
                
                # half of positive input images are hard positives, ie no stitching
                else: 
                    img_comb_free = img_original
                    watermarked_comb_mask = watermarked_img
                
                # solve for binary masks
                watermarked_mask = solve_mask(watermarked_comb_mask, img_comb_free)

                '''saving'''
                save_id = f'{i}.jpg'
                cv2.imwrite(osp.join(img_input_path, save_id),
                            cv2.cvtColor(np.array(watermarked_comb_mask.permute(1, 2, 0) * 255), cv2.COLOR_BGR2RGB))

                cv2.imwrite(osp.join(watermark_mask_path, save_id),
                            np.concatenate((watermarked_mask[:, :, np.newaxis],
                                            watermarked_mask[:, :, np.newaxis],
                                            watermarked_mask[:, :, np.newaxis]), 2) * 256.0)

                i += 1


if __name__ == "__main__":
    main()
