
import os
import pandas
import cv2
import numpy as np
import torchvision
from typing import Optional, List
import tqdm


class So2SatDatasetCustom(torchvision.datasets.VisionDataset):
    """
    So2Sat_POP dataset loader with custom normalization.
    
    Custom features:
    - Uses RGB bands (2-4) from sen2spring Sentinel-2 images
    - Custom image normalization: clip to [0, 4000], divide by 4000, then min-max normalize
    - Custom label normalization with specified mean and std
    - Optional preloading: load all images into RAM at init to avoid repeated disk I/O
    
    Dataset structure:
    - Part1/train/city_name/city_name.csv (contains GRD_ID, Class, POP)
    - Part1/train/city_name/sen2spring/Class_X/GRD_ID_sen2spring.tif (Sentinel-2 multi-spectral images)
    
    This class expects a FileList.csv in the root directory with columns:
    FileName, SPLIT, POP (or target column name), SSL_SPLIT (optional)
    
    Sentinel-2 images have 13 spectral channels. RGB bands are extracted from channels [3, 2, 1] (bands 2-4).
    """
    
    def __init__(self, root=None,
                 split="train", 
                 target_type="POP",
                 mean=0., 
                 std=1.,
                 pad=None,
                 ssl_type=0,
                 ssl_postfix="",
                 ssl_mult=1,
                 image_dir="So2Sat_POP_Part1",  # Part1 contains Sentinel-2 images
                 file_list_name="FileList.csv",
                 normalize_mean=1085.0,  # Label normalization mean
                 normalize_std=2800.0,   # Label normalization std
                 preload=False           # If True, preload all images into RAM at init
                 ):
        if root is None:
            raise ValueError("root value is required")
        
        super().__init__(root)
        
        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.pad = pad
        self.ssl_type = ssl_type
        self.ssl_postfix = ssl_postfix
        self.ssl_mult = ssl_mult
        self.image_dir = image_dir
        self.file_list_name = file_list_name
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.preload = preload
        
        self.fnames, self.outcome = [], []
        self.preloaded_images = None
        
        # Load file list
        file_list_path = os.path.join(self.root, "{}{}.csv".format(self.file_list_name.replace(".csv", ""), self.ssl_postfix))
        print("Using data file from ", file_list_path)
        
        if not os.path.exists(file_list_path):
            raise FileNotFoundError(f"FileList not found: {file_list_path}. Please create FileList.csv first.")
        
        with open(file_list_path) as f:
            data = pandas.read_csv(f)
        
        if "SPLIT" in data.columns:
            data["SPLIT"] = data["SPLIT"].map(lambda x: str(x).upper())
        
        if len(self.ssl_postfix) > 0:
            data_train_lab = data[(data["SPLIT"] == "TRAIN") & (data["SSL_SPLIT"] == "LABELED")].copy()
        else:
            data_train_lab = data[(data["SPLIT"] == "TRAIN")].copy() if "SPLIT" in data.columns else data
        
        if self.split != "ALL" and "SPLIT" in data.columns:
            data = data[data["SPLIT"] == self.split]
        
        # Handle SSL splits
        if self.ssl_type == 1:
            assert self.split == "TRAIN", "subset selection only for train"
            if "SSL_SPLIT" in data.columns:
                data = data[data["SSL_SPLIT"] == "LABELED"]
            print("Using SSL_SPLIT Labeled, total samples", len(data))
            data_columns = data.columns
            if self.ssl_mult < 0:
                data = pandas.DataFrame(np.repeat(data.values, 2, axis=0))
            else:
                data = pandas.DataFrame(np.repeat(data.values, self.ssl_mult, axis=0))
            data.columns = data_columns
            print("data after duplicates:", len(data))
        
        elif self.ssl_type == 2:
            assert self.split == "TRAIN", "subset selection only for train"
            if "SSL_SPLIT" in data.columns:
                data = data[data["SSL_SPLIT"] != "LABELED"]
            print("Using SSL_SPLIT unlabeled, total samples", len(data))
        
        elif self.ssl_type == 0:
            print("Using SSL_SPLIT ALL, total samples", len(data))
        
        self.header = data.columns.tolist()
        self.fnames = data["FileName"].tolist()
        self.outcome = data.values.tolist()
        
        # Verify files exist
        missing = []
        for fname in self.fnames:
            full_path = os.path.join(self.root, fname)
            if not os.path.exists(full_path):
                missing.append(fname)
        
        if len(missing) > 0:
            print("{} images could not be found:".format(len(missing)))
            for f in sorted(missing)[:10]:  # Show first 10
                print("\t", f)
            if len(missing) > 10:
                print(f"\t... and {len(missing) - 10} more")
            raise FileNotFoundError(f"Missing files. First missing: {sorted(missing)[0]}")
        
        # Preload all images into memory if requested
        if self.preload:
            print(f"Preloading {len(self.fnames)} images into memory...")
            self.preloaded_images = []
            for idx in tqdm.tqdm(range(len(self.fnames)), desc=f"Preloading {self.split}"):
                img = self._load_and_process_image(idx)
                self.preloaded_images.append(img)
            print(f"✓ Preloaded {len(self.preloaded_images)} images (~{len(self.preloaded_images) * 3 * 224 * 224 * 4 / (1024**3):.2f} GB)")
    
    def _load_and_process_image(self, index):
        """Load and process a single image from disk (without augmentation)."""
        image_path = os.path.join(self.root, self.fnames[index])
        
        # Load image - handle Sentinel-2 multi-spectral TIFF files
        if image_path.endswith('.tif') or image_path.endswith('.tiff'):
            # Check if this is a Sentinel-2 file (sen2spring, sen2summer, etc.)
            is_sentinel2 = 'sen2' in image_path.lower()
            
            if is_sentinel2:
                # Use tifffile for multi-spectral Sentinel-2 data (13 channels)
                try:
                    import tifffile
                    # Read the image
                    data = tifffile.imread(image_path)
                    
                    # Sentinel-2 shape is (H, W, 13) - extract RGB bands
                    # Band 4 (index 3) = Red, Band 3 (index 2) = Green, Band 2 (index 1) = Blue
                    if len(data.shape) == 3 and data.shape[2] >= 4:
                        # Extract RGB: Red (index 3), Green (index 2), Blue (index 1)
                        image_bands = data[:, :, [3, 2, 1]].astype(np.float32)  # Shape: (H, W, 3) - RGB order
                        
                        # Apply normalization: Clip to [0, 4000] and min-max normalize
                        # Step 1: Clip to [0, 4000]
                        image_bands = np.clip(image_bands, 0, 4000)
                        
                        # Step 2: Min-max normalization to [0, 1]
                        arr_min = image_bands.min()
                        arr_max = image_bands.max()
                        if arr_max > arr_min:
                            image_bands = (image_bands - arr_min) / (arr_max - arr_min)
                        
                        # Transpose to (C, H, W)
                        photo = image_bands.transpose((2, 0, 1)).copy()
                    else:
                        raise ValueError(f"Unexpected Sentinel-2 image shape: {data.shape}")
                except ImportError:
                    raise ImportError("tifffile is required for Sentinel-2 images. Install with: pip install tifffile")
                except Exception as e:
                    raise ValueError(f"Failed to load image {image_path}: {e}")
            else:
                # Standard TIFF files (DEM, etc.)
                photo = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                if photo is None:
                    raise ValueError(f"Failed to load image: {image_path}")
                if len(photo.shape) == 3:
                    photo = photo.transpose((2, 0, 1)).copy()
                elif len(photo.shape) == 2:
                    photo = photo[np.newaxis, :, :].copy()
            
            # Resize to 224x224 for model compatibility (EfficientNet/ResNet expect 224x224)
            if photo is not None and photo.size > 0:
                # Resize each channel separately
                c, h, w = photo.shape
                photo_resized = np.zeros((c, 224, 224), dtype=photo.dtype)
                for i in range(c):
                    photo_resized[i] = cv2.resize(photo[i], (224, 224), interpolation=cv2.INTER_LINEAR)
                photo = photo_resized.copy()
            else:
                raise ValueError(f"Failed to process image: {image_path}")
        else:
            # Standard image formats (JPEG, PNG, etc.)
            photo = cv2.imread(image_path).astype(np.float32)
            # Resize to 224x224 for model compatibility
            if photo is not None and photo.size > 0:
                photo = cv2.resize(photo, (224, 224), interpolation=cv2.INTER_LINEAR)
            else:
                raise ValueError(f"Failed to load image: {image_path}")
            photo = photo.transpose((2, 0, 1)).copy()  # Make contiguous after transpose
        
        # Ensure 3 channels for RGB models (if single channel, repeat)
        if photo.shape[0] == 1:
            photo = np.repeat(photo, 3, axis=0).copy()
        elif photo.shape[0] > 3:
            photo = photo[:3, :, :].copy()
        
        return photo
    
    def __getitem__(self, index):
        # Load image (either from preloaded memory or from disk)
        if self.preload and self.preloaded_images is not None:
            photo = self.preloaded_images[index].copy()
        else:
            photo = self._load_and_process_image(index)
        
        # Data augmentation: random horizontal flip
        if np.random.randint(0, 2) == 0:
            photo = photo[:, :, ::-1].copy()
        
        # Gather targets
        target = []
        for t in self.target_type:
            if t == "Filename":
                target.append(self.fnames[index])
            else:
                # Get raw target value (no normalization here - done in training loop)
                raw_target = np.float32(self.outcome[index][self.header.index(t)])
                target.append(raw_target)
        
        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
        
        # Apply padding augmentation if specified
        if self.pad is not None:
            c, h, w = photo.shape
            temp1 = np.zeros((c, h + 2 * self.pad, w + 2 * self.pad), dtype=photo.dtype)
            temp1[:, self.pad:-self.pad, self.pad:-self.pad] = photo
            i1, j1 = np.random.randint(0, 2 * self.pad, 2)
            photo = temp1[:, i1:(i1 + h), j1:(j1 + w)].copy()
        
        # Ensure contiguous and independent array
        photo_final = np.ascontiguousarray(photo)
        
        return photo_final, target
    
    def __len__(self):
        return len(self.fnames)
    
    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "SPLIT: {split}"]
        return '\n'.join(lines).format(**self.__dict__)
