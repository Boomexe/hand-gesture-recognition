import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from gesture_preprocessing import add_velocity

class HDF5GestureDataset(Dataset):
    def __init__(self, path, numbered_unique_words, should_target_specific_words=False, target_words_set=None):
        self.path = path
        self.indices = []
        self.h5 = None
        self.should_target_specific_words = should_target_specific_words
        self.target_words_set = set(target_words_set) if target_words_set is not None else set()
        self.numbered_unique_words = numbered_unique_words

        with h5py.File(self.path, "r") as h5:
            total = int(h5.attrs["num_samples"])

            for i in range(total):
                word = h5[str(i)].attrs["word"]

                if not self.should_target_specific_words or word in self.target_words_set:
                    self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["h5"] = None
        return state

    def _get_file(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.path, "r")

        return self.h5

    def __getitem__(self, index):
        h5 = self._get_file()

        sample_index = self.indices[index]
        group = h5[str(sample_index)]

        gesture = group["data"][:].astype(np.float32, copy=False)
        word = group.attrs["word"]

        gesture = add_velocity(gesture[np.newaxis, ...])[0]

        x = torch.from_numpy(gesture)
        y = torch.tensor(self.numbered_unique_words[word], dtype=torch.long)
        
        return x, y