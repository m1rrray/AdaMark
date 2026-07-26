"""Datasets: FFHQ covers for training and tamper-localization datasets for evaluation"""

import glob
import os
import random

from PIL import Image
from torch.utils.data import Dataset


def create_tamper_dataset(dataset_name, base_dir, transform, limit=None, tamper_filter=None):
    """Factory for the supported tamper-localization datasets"""

    dataset_name = dataset_name.lower()

    if dataset_name == "casia1":
        return Casia1Dataset(
            au_dir=f"{base_dir}/Au",
            modified_tp_dir=f"{base_dir}/ModifiedTP",
            gt_dir=f"{base_dir}/groundtruth",
            transform=transform,
            limit=limit,
            tamper_filter=tamper_filter,
        )

    if dataset_name == "casia2":
        return Casia2Dataset(
            au_dir=f"{base_dir}/Au",
            tp_dir=f"{base_dir}/Tp",
            gt_dir=f"{base_dir}/Groundtruth",
            transform=transform,
            limit=limit,
            tamper_filter=tamper_filter,
        )

    if dataset_name == "columbia":
        return ColumbiaDataset(
            auth_dir=f"{base_dir}/authentic",
            splice_dir=f"{base_dir}/forged",
            gt_dir=f"{base_dir}/masks",
            transform=transform,
            limit=limit,
        )

    raise ValueError(f"Unknown dataset: {dataset_name}")


class ImageDataset(Dataset):
    """Flat directory of cover images such as FFHQ, yields a cover tensor and None"""

    def __init__(self, cover_dir, transform):
        self.cover_dir = cover_dir
        self.transform = transform

        self.cover_files = sorted(os.listdir(cover_dir))

    def __len__(self):
        return len(self.cover_files)

    def __getitem__(self, idx):
        cover_path = os.path.join(self.cover_dir, self.cover_files[idx])
        cover_image = Image.open(cover_path).convert("RGB")

        return self.transform(cover_image), None


class Casia2Dataset(Dataset):
    """CASIA2 dataset, yields an authentic image, a forged image and a GT mask"""

    def __init__(self, au_dir, tp_dir, gt_dir, transform, limit=None, tamper_filter=None):
        self.gt_dir = gt_dir
        self.transform = transform

        all_tp_files = sorted(glob.glob(os.path.join(tp_dir, "*.jpg")) +
                              glob.glob(os.path.join(tp_dir, "*.tif")))

        self.tp_files = []
        for f in all_tp_files:
            info = self.parse_tp_name(f)
            if tamper_filter is None or info["type"] == tamper_filter:
                self.tp_files.append(f)

        if limit:
            self.tp_files = random.sample(self.tp_files, min(limit, len(self.tp_files)))

        self.au_dict = {}
        for path in glob.glob(os.path.join(au_dir, "*.*")):
            name = os.path.basename(path)
            parts = name.split("_")
            if len(parts) >= 3:
                au_id = parts[1] + parts[2].split(".")[0]
                self.au_dict[au_id] = path

        filter_name = tamper_filter if tamper_filter else "all types"
        print(f"[{filter_name.upper()}] Initialized: {len(self.tp_files)} forgeries, "
              f"{len(self.au_dict)} originals.")

    def __len__(self):
        return len(self.tp_files)

    def parse_tp_name(self, filename):
        name = os.path.basename(filename).split(".")[0]
        parts = name.split("_")
        return {
            "type": "splicing" if parts[1] == "D" else "copymove",
            "target_id": parts[5] if len(parts) > 7 else None,
            "full_name": name,
        }

    def __getitem__(self, idx):
        tp_path = self.tp_files[idx]
        info = self.parse_tp_name(tp_path)

        tp_img = Image.open(tp_path).convert("RGB")

        au_path = self.au_dict.get(info["target_id"])
        if au_path and os.path.exists(au_path):
            au_img = Image.open(au_path).convert("RGB")
        else:
            au_img = tp_img.copy()

        gt_path = os.path.join(self.gt_dir, info["full_name"] + "_gt.png")
        if os.path.exists(gt_path):
            mask_img = Image.open(gt_path).convert("L")
        else:
            mask_img = Image.new("L", tp_img.size, 0)

        au_img = self.transform(au_img)
        tp_img = self.transform(tp_img)
        mask_tensor = (self.transform(mask_img) > 0.5).float()

        return au_img, tp_img, mask_tensor


class Casia1Dataset(Dataset):
    """CASIA1 dataset, yields an authentic image, a forged image and a GT mask"""

    def __init__(self, au_dir, modified_tp_dir, gt_dir, transform, limit=None, tamper_filter=None):
        self.transform = transform

        self.gt_dict = {}
        for sub in ["Sp", "CM"]:
            for gt_path in glob.glob(os.path.join(gt_dir, sub, "*_gt.png")):
                base_name = os.path.basename(gt_path).replace("_gt.png", "")
                self.gt_dict[base_name] = gt_path

        self.au_dict = {}
        for path in glob.glob(os.path.join(au_dir, "Au_*.*")):
            name = os.path.basename(path)
            parts = name.split("_")
            if len(parts) >= 3:
                au_id = f"{parts[1]}{parts[2].split('.')[0]}"
                self.au_dict[au_id] = path

        all_tp_files = []
        for sub in ["Sp", "CM"]:
            pattern = os.path.join(modified_tp_dir, sub, "*.*")
            all_tp_files.extend(glob.glob(pattern))

        self.tp_files = []
        for tp_path in sorted(all_tp_files):
            info = self.parse_tp_name(tp_path)
            has_au = info["target_id"] in self.au_dict
            has_gt = info["full_name"] in self.gt_dict
            if has_au and has_gt:
                if tamper_filter is None or info["type"] == tamper_filter:
                    self.tp_files.append(tp_path)

        if limit:
            self.tp_files = random.sample(self.tp_files, min(limit, len(self.tp_files)))

        filter_name = tamper_filter if tamper_filter else "all types"
        print(f"[{filter_name.upper()}] Raw files found: {len(all_tp_files)}")
        print(f"[{filter_name.upper()}] Complete triplets kept: {len(self.tp_files)}")

    def parse_tp_name(self, filename):
        name = os.path.basename(filename).split(".")[0]
        parts = name.split("_")
        target_id = parts[4] if len(parts) > 5 else None
        parent_dir = os.path.basename(os.path.dirname(filename))
        t_type = "splicing" if parent_dir == "Sp" else "copymove"
        return {"full_name": name, "target_id": target_id, "type": t_type}

    def __getitem__(self, idx):
        tp_path = self.tp_files[idx]
        info = self.parse_tp_name(tp_path)

        tp_img = Image.open(tp_path).convert("RGB")
        au_img = Image.open(self.au_dict[info["target_id"]]).convert("RGB")
        mask_img = Image.open(self.gt_dict[info["full_name"]]).convert("L")

        au_img = self.transform(au_img)
        tp_img = self.transform(tp_img)
        mask_tensor = (self.transform(mask_img) > 0.5).float()

        return au_img, tp_img, mask_tensor

    def __len__(self):
        return len(self.tp_files)


class ColumbiaDataset(Dataset):
    """Columbia splicing dataset, yields an authentic image, a spliced image and a mask"""

    def __init__(self, auth_dir, splice_dir, gt_dir, transform, limit=None):
        self.transform = transform

        self.auth_paths = sorted(glob.glob(os.path.join(auth_dir, "*.*")))
        self.splice_paths = sorted(glob.glob(os.path.join(splice_dir, "*.*")))
        self.gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.*")))

        if len(self.gt_paths) == 0:
            raise ValueError(f"No masks found in directory: {gt_dir}")

        if limit is not None:
            self.auth_paths = self.auth_paths[:limit]

    def __len__(self):
        return len(self.auth_paths)

    def __getitem__(self, idx):
        auth_path = self.auth_paths[idx]
        splice_path = self.splice_paths[idx % len(self.splice_paths)]
        gt_path = random.choice(self.gt_paths)

        cover_img = Image.open(auth_path).convert("RGB")
        tampered_img = Image.open(splice_path).convert("RGB")
        mask_img = Image.open(gt_path).convert("RGB")

        cover_img = self.transform(cover_img)
        tampered_img = self.transform(tampered_img)

        # Columbia masks are colour-coded; the green channel carries the region.
        mask_tensor = (self.transform(mask_img)[1] > 0.5).float().unsqueeze(0)

        return cover_img, tampered_img, mask_tensor
