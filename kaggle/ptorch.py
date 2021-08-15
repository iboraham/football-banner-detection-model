import json
from torch.utils.data import Dataset, DataLoader
from skimage import io
import pandas as pd
import torch
import os
import numpy as np
from pathlib import Path
import cv2, zlib, base64
from PIL import Image

DATA_PATH = "../input/football-advertising-banners-detection/football"
meta_class_data = {
    "mastercard": 0,
    "nissan": 1,
    "playstation": 2,
    "unicredit": 3,
    "pepsi": 4,
    "adidas": 5,
    "gazprom": 6,
    "heineken": 7,
}


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


class FootballBannerDataset(Dataset):
    """Football advertising banners images from UEFA Champions League matches."""

    def __init__(self, annotations_dir: str, root_dir: str, transform=None) -> None:
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            annotations_dir (string): Directory with all the annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations_dir = annotations_dir
        self.get_banner_frame()
        self.root_dir = root_dir
        self.transform = transform

    def get_banner_frame(self):
        annot_list = list(Path(self.annotations_dir).glob("*.json"))
        banner_list = []
        for annot_file in annot_list:
            with annot_file.open("r", encoding="utf-8") as annotReader:
                labels = json.loads(annotReader.read())
            if len(labels["objects"]) == 0:
                continue
            image_file = os.path.splitext(annot_file)[0].split("/")[-1]
            bitmap = labels["objects"][0]["bitmap"]["data"]
            origin = labels["objects"][0]["bitmap"]["origin"]
            label = meta_class_data[labels["objects"][0]["classTitle"]]
            banner_list.append([image_file, bitmap, origin[0], origin[1], label])
        self.banner_frame = pd.DataFrame(
            banner_list, columns=["image_file", "bitmap", "origin0", "origin1", "label"]
        )

    def __len__(self):
        return len(self.banner_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.banner_frame.iloc[idx, 0])
        image = io.imread(img_name)
        mask_small = base64_2_mask(self.banner_frame.iloc[idx, 1])
        start_point = [self.banner_frame.iloc[idx, 2], self.banner_frame.iloc[idx, 3]]
        crImage = image[
            start_point[1] : start_point[1] + mask_small.shape[0],
            start_point[0] : start_point[0] + mask_small.shape[1],
        ]
        dim = (160, 90)

        # resize image
        crImage = cv2.resize(crImage, dim, interpolation=cv2.INTER_AREA)
        label = self.banner_frame.iloc[idx, 4]
        sample = {"image": crImage.transpose((0, 3, 1, 2)), "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample
