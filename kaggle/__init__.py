from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import torch
import torchvision
import json
from pathlib import Path
import pathlib
import numpy as np
import cv2, zlib, base64, io
from PIL import Image
import json
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask


def mask_2_base64(mask):
    img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
    img_pil.putpalette([0, 0, 0, 255, 255, 255])
    bytes_io = io.BytesIO()
    img_pil.save(bytes_io, format="PNG", transparency=0, optimize=0)
    bytes = bytes_io.getvalue()
    return base64.b64encode(zlib.compress(bytes)).decode("utf-8")


def lookImageFile(
    image_file_list: list, match_annots_name: str, images_path: pathlib.PosixPath
) -> np.ndarray:
    """
    Looks file images folder, match with given file name and return reeaded image
    file as numpy.ndarray.

    Args:
        image_file_list (list): images list from pathlib. Input should be a Python list type.
        images_path (pathlib.PosixPath):
        match_annots_name (str): Image name as string to match.

    Returns:
        np.ndarray: Result of image after readed.

    Example usage:
        >> from pathlib import Path
        >> images_path = Path("input/images")
        >> image_file_list = list(.glob(r"*.png"))
        >> lookImageFile(
            image_file_list,
            match_annots_name = "input/annotations/example.json",
            images_path = images_path
            )
    """
    match_image_name = os.path.splitext(match_annots_name)[0].split("/")[-1]
    image_file = images_path.joinpath(match_image_name)
    try:
        image_file_list.remove(image_file)
    except ValueError:
        return None
    img = cv2.imread(str(image_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def create_mask(image, labels):
    try:
        mask_str = labels["objects"][0]["bitmap"]["data"]
    except IndexError:
        if len(labels["objects"]) == 0:
            return None
    mask_small = base64_2_mask(mask_str)
    mask = np.zeros(image.shape[:2], dtype="uint8")
    start_point = labels["objects"][0]["bitmap"]["origin"]
    mask[
        start_point[1] : start_point[1] + mask_small.shape[0],
        start_point[0] : start_point[0] + mask_small.shape[1],
    ] = mask_small

    return mask


def create_little_image(image, labels):
    try:
        mask_str = labels["objects"][0]["bitmap"]["data"]
    except IndexError:
        if len(labels["objects"]) == 0:
            return None
    mask_small = base64_2_mask(mask_str)
    start_point = labels["objects"][0]["bitmap"]["origin"]
    cropped_image = image[
        start_point[1] : start_point[1] + mask_small.shape[0],
        start_point[0] : start_point[0] + mask_small.shape[1],
    ]

    return cropped_image, mask_small


def create_env():
    DATA_PATH = Path("../input/football-advertising-banners-detection/football")
    annot_path = DATA_PATH.joinpath("annotations")
    images_path = DATA_PATH.joinpath("images")
    annots = list(annot_path.glob(r"*.json"))
    images = list(images_path.glob(r"*.png"))
    return images_path, annots, images


images_path, annots, images = create_env()

# read json
data = []

for annot_fileJSON in tqdm(annots[1]):
    with annot_fileJSON.open("r", encoding="utf-8") as jsonReader:
        labels = json.loads(jsonReader.read())

    image = lookImageFile(images, annot_fileJSON, images_path)
    if type(image) != np.ndarray:
        continue

    image, mask = create_mask(image, labels)
    if type(mask) != np.ndarray:
        continue

    res = cv2.bitwise_and(image, image, mask=mask)
    plt.imshow(res)
    data.append(res.flatten())
df = pd.DataFrame(data)
