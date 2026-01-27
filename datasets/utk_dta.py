

import os
import collections
import pandas
import datetime
import cv2
import re

import numpy as np
import skimage.draw
import torchvision

class UTKdta(torchvision.datasets.VisionDataset):
    
    @staticmethod
    def extract_age_from_filename(filename):
        """
        Extract age from UTKFace filename.
        Format: age_gender_race_date.jpg.chip.jpg
        Returns the age as integer, or None if extraction fails.
        """
        basename = os.path.basename(filename)
        # Extract first number before first underscore
        match = re.match(r'^(\d+)_', basename)
        if match:
            return int(match.group(1))
        else:
            return None

    def __init__(self, root=None,
                 split="train", target_type="age",
                 mean=0., std=1.,
                 pad=None,
                 ssl_type = 0,
                 ssl_postfix = "",
                 ssl_mult = 1,
                 image_dir = "UTKFace",  # Configurable image directory
                 file_list_name = "FileList.csv"  # Configurable file list name
                 ):
        if root is None:
            assert 1==2, "need root value"

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

        self.fnames, self.outcome = [], []

        # Load photo-level labels
        file_list_path = os.path.join(self.root, "{}{}.csv".format(self.file_list_name.replace(".csv", ""), self.ssl_postfix))
        print("Using data file from ", file_list_path)

        with open(file_list_path) as f:
            data = pandas.read_csv(f)
        data["SPLIT"].map(lambda x: x.upper())

        if len(self.ssl_postfix) > 0:
            data_train_lab = data[(data["SPLIT"] == "TRAIN") & (data["SSL_SPLIT"] == "LABELED")].copy()
        else:
            data_train_lab = data[(data["SPLIT"] == "TRAIN")].copy()

        if self.split != "ALL":
            data = data[data["SPLIT"] == self.split]


        if self.ssl_type == 1:
            assert self.split == "TRAIN", "subset selection only for train"
            data = data[data["SSL_SPLIT"] == "LABELED"]
            print("Using SSL_SPLIT Labeled, total samples", len(data))
            data_columns = data.columns
            if self.ssl_mult < 0:
                data = pandas.DataFrame(np.repeat(data.values,2,axis=0))
            else:
                data = pandas.DataFrame(np.repeat(data.values,self.ssl_mult,axis=0))
            data.columns = data_columns
            print("data after duplicates:", len(data))

        elif self.ssl_type == 2:
            assert self.split == "TRAIN", "subset selection only for train"
            data = data[data["SSL_SPLIT"] != "LABELED"]
            print("Using SSL_SPLIT unlabeled, total samples", len(data))
            data_columns = data.columns
            print("data after duplicates:", len(data))

        elif self.ssl_type == 0:
            print("Using SSL_SPLIT ALL, total samples", len(data))
            pass
        else:
            assert 1==2, "invalid option for ssl_type"


        self.header = data.columns.tolist()
        self.fnames = data["FileName"].tolist()

        self.outcome = data.values.tolist()

        missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, self.image_dir)))
        if len(missing) != 0:
            print("{} photos could not be found in {}:".format(len(missing), os.path.join(self.root, self.image_dir)))
            for f in sorted(missing):
                print("\t", f)
            raise FileNotFoundError(os.path.join(self.root, self.image_dir, sorted(missing)[0]))

    def __getitem__(self, index):

        photo_path = os.path.join(self.root, self.image_dir, self.fnames[index])
        photo = cv2.imread(photo_path).astype(np.float32)
        # Resize image to fixed size (224x224) for ResNet compatibility
        if photo is not None and photo.size > 0:
            photo = cv2.resize(photo, (224, 224), interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError(f"Failed to load image: {photo_path}")
        photo = photo.transpose((2, 0, 1))


        # Apply normalization
        if isinstance(self.mean, (float, int)):
            photo -= self.mean
        else:
            photo -= self.mean.reshape(3, 1, 1)

        if isinstance(self.std, (float, int)):
            photo /= self.std
        else:
            photo /= self.std.reshape(3, 1, 1)

        # Set number of frames
        c, h, w = photo.shape
        
        if np.random.randint(0,2) == 0:
            photo = photo[:,:,::-1]

        # Gather targets
        target = []
        for t in self.target_type:
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "age":
                # For age, extract from filename instead of CSV (more reliable)
                age_from_filename = self.extract_age_from_filename(self.fnames[index])
                if age_from_filename is not None:
                    target.append(np.float32(age_from_filename))
                else:
                    # Fallback to CSV if filename extraction fails
                    if t in self.header:
                        target.append(np.float32(self.outcome[index][self.header.index(t)]))
                    else:
                        raise ValueError(f"Could not extract age from filename {self.fnames[index]} and 'age' not in CSV header")
            else:
                target.append(np.float32(self.outcome[index][self.header.index(t)]))

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
        
        if self.pad is not None:

            photo1 = photo.copy()

            c, h, w = photo.shape

            temp1 = np.zeros((c, h + 2 * self.pad, w + 2 * self.pad), dtype=photo.dtype)
            temp1[:, self.pad:-self.pad, self.pad:-self.pad] = photo1

            i1, j1 = np.random.randint(0, 2 * self.pad, 2)
            photo1 = temp1[:, i1:(i1 + h), j1:(j1 + w)]

        else:
            photo1 = photo.copy()
            i1 = 0
            j1 = 0

        return photo1, target

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "SPLIT: {split}"]
        return '\n'.join(lines).format(**self.__dict__)











def _defaultdict_of_lists():

    return collections.defaultdict(list)





