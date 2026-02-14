import os
import json
from PIL import Image
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# Configuration for dataset cache
CACHE_DIR = "/root/autodl-tmp/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

class Flickr30kDataLoader:
    def __init__(self, split="test", batch_size=32):
        self.split = split
        self.batch_size = batch_size
        self.dataset = None
        self.dataloader = None
    
    def load_data(self):
        # Load Flickr30k Dataset
        self.dataset = load_dataset("nlphuji/flickr30k", cache_dir=CACHE_DIR)
        
        # Filter dataset based on the provided split
        self.dataset = self.dataset['test'].filter(lambda example: example['split'] == self.split)
        print(f"Filtered {self.split} set size: {len(self.dataset)}")
        
    def collate_fn(self, batch):

        images = [item["image"] for item in batch]
        captions = [item["caption"] for item in batch]
        return {"images": images, "captions": captions}
    
    def get_dataloader(self):
        if self.dataset is None:
            self.load_data()
        
        # Create DataLoader
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
        return self.dataloader


class MSCOCODataLoader:
    def __init__(self, split="test", batch_size=32):
        self.split = split
        self.batch_size = batch_size
        self.dataset = None
        self.dataloader = None

    def load_data(self):
        # Load MSCOCO Captions Dataset
        self.dataset = load_dataset("clip-benchmark/wds_mscoco_captions", cache_dir=CACHE_DIR)

        # Access the required split
        if self.split in self.dataset:
            self.dataset = self.dataset[self.split]
        else:
            raise ValueError(f"Invalid split '{self.split}'. Available splits are: {list(self.dataset.keys())}")

        print(f"Loaded {self.split} set size: {len(self.dataset)}")

    def collate_fn(self, batch):
        images = [item["jpg"] for item in batch]
        captions = [item["txt"].split('\n') for item in batch]
        return {"images": images, "captions": captions}

    def get_dataloader(self):
        if self.dataset is None:
            self.load_data()

        # Create DataLoader (no shuffling needed for testing)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn
        )
        return self.dataloader


class HatefulMemesDataLoader:
    def __init__(self, data_root, split="dev_seen", batch_size=32):
        self.data_root = data_root
        self.split = split
        self.batch_size = batch_size
        self.data = []
        self.dataloader = None
    
    def load_data(self):
        # Load jsonl file
        jsonl_path = os.path.join(self.data_root, self.split + ".jsonl")
        print(f"Loading data from {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        print(f"Loaded {len(self.data)} samples.")

    def collate_fn(self, batch):
        images = []
        texts = [] # Raw text of the meme
        labels = []
        ids = []
        
        for item in batch:
            # Load image
            # item["img"] is like "img/12345.png"
            # self.data_root is "C:/.../datasets/hateful_memes"
            img_path = os.path.join(self.data_root, item["img"])
            # Normalize path separators for Windows
            img_path = os.path.normpath(img_path)
            
            try:
                if not os.path.exists(img_path):
                     # Try to see if img folder is missing in path or similar
                     # Sometimes datasets have structure `data/img` but json says `img/01.png`
                     pass
                
                image = Image.open(img_path).convert("RGB")
                images.append(image)
                texts.append(item["text"])
                labels.append(item["label"])
                ids.append(item["id"])
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
                
        return {
            "images": images, 
            "texts": texts, 
            "labels": torch.tensor(labels),
            "ids": ids
        }
    
    def get_dataloader(self):
        if not self.data:
            self.load_data()
        
        self.dataloader = DataLoader(
            self.data, 
            batch_size=self.batch_size, 
            collate_fn=self.collate_fn,
            shuffle=False
        )
        return self.dataloader





class WinogroundDataLoader:
    def __init__(self, auth_token=None, batch_size=32):
        self.auth_token = auth_token
        self.batch_size = batch_size
        self.dataset = None
        self.dataloader = None

    def load_data(self):
        try:
            self.dataset = load_dataset("facebook/winoground", split="test", use_auth_token=self.auth_token, cache_dir=CACHE_DIR)
        except Exception as e:
            print(f"Error loading Winoground: {e}")
            raise e
        print(f"Loaded Winoground size: {len(self.dataset)}")

    def collate_fn(self, batch):
        images = []
        captions = []
        for item in batch:
            images.append(item["image_0"].convert("RGB"))
            captions.append([item["caption_0"]])
            
            images.append(item["image_1"].convert("RGB"))
            captions.append([item["caption_1"]])
            
        return {"images": images, "captions": captions}

    def get_dataloader(self):
        if self.dataset is None:
            self.load_data()
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size // 2, collate_fn=self.collate_fn)
        return self.dataloader


class VizWizDataLoader:
    def __init__(self, split="val", batch_size=32):
        self.split = split
        self.batch_size = batch_size
        self.dataset = None
        self.dataloader = None

    def load_data(self):
        # Load VizWiz-Captions (VizWiz-Caps)
        self.dataset = load_dataset("lmms-lab/VizWiz-Caps", split=self.split, cache_dir=CACHE_DIR)
        self.has_captions = True
            
        print(f"Loaded VizWiz {self.split} size: {len(self.dataset)}")

    def collate_fn(self, batch):
        images = []
        captions = []
        for item in batch:
            images.append(item["image"].convert("RGB"))
            if self.has_captions:
                # VizWiz-Captions entries
                caps = item.get("text_annotations", item.get("caption", [""]))
                if isinstance(caps, str): caps = [caps]
                captions.append(caps if caps else [""])
            else:
                # VQA entries, use question as text
                q = item.get("question", "")
                captions.append([q] if q else [""])
        
        return {"images": images, "captions": captions}

    def get_dataloader(self):
        if self.dataset is None:
            self.load_data()
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
        return self.dataloader


class CIFAR100DataLoader:
    def __init__(self, split="test", batch_size=100, one_image_per_class=False):
        self.split = split
        self.batch_size = batch_size
        self.one_image_per_class = one_image_per_class
        self.dataset = None
        self.dataloader = None
        self.classes = None

    def load_data(self):
        self.dataset = load_dataset("cifar100", split=self.split, cache_dir=CACHE_DIR)
        self.classes = self.dataset.features["fine_label"].names
        
        if self.one_image_per_class:
            indices = []
            seen_labels = set()
            # Iterate through the dataset to find the first image for each class
            # Note: This might be slow if the dataset is huge, but CIFAR test is 10k, so it's fast.
            for idx in range(len(self.dataset)):
                label = self.dataset[idx]["fine_label"]
                if label not in seen_labels:
                    seen_labels.add(label)
                    indices.append(idx)
                if len(seen_labels) == 100:
                    break
            self.dataset = self.dataset.select(indices)
            print(f"Filtered CIFAR-100 to one image per class. Size: {len(self.dataset)}")
        else:
            print(f"Loaded CIFAR-100 {self.split} size: {len(self.dataset)}")

    def collate_fn(self, batch):
        images = []
        captions = []
        for item in batch:
            images.append(item["img"].convert("RGB"))
            label_idx = item["fine_label"]
            classname = self.classes[label_idx]
            captions.append([f"a photo of a {classname}"])
            
        return {
            "images": images,
            "captions": captions,
        }

    def get_dataloader(self):
        if self.dataset is None:
            self.load_data()
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=False)
        return self.dataloader


# utility functions

def dataloader_with_indices(dataloader):
    start = 0
    for batch in dataloader:
        end = start + len(batch['images'])
        inds = torch.arange(start, end)
        yield batch['images'], batch['captions'], inds
        start = end

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)

