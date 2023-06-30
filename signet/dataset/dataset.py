import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocess import Preprocess
class SignLanguageDataset(Dataset):
    def __init__(self, npy_file_paths, labels,CFG):
        assert len(npy_file_paths) == len(labels), "Mismatch between number of .npy files and labels"
        self.npy_file_paths = npy_file_paths
        self.labels = labels
        self.preprocess = Preprocess(CFG)
        self.char_to_idx = CFG.char_to_idx
        self.max_wordlen = CFG.MAX_WORD_LENGTH
        self.max_framelen = CFG.max_len

        self.word_pad = CFG.PAD[1]
        self.frames_pad = CFG.PAD[0]

    def __len__(self):
        return len(self.npy_file_paths)

    def __getitem__(self, idx):
        npy_file = np.load(self.npy_file_paths[idx])

        # Padding
        pad_length = max(0, self.frames_pad - npy_file.shape[0])
        npy_file = np.pad(npy_file, ((0, pad_length), (0, 0), (0, 0)), mode='constant')

        # Ensure the numpy array is of the correct shape 
        assert npy_file.shape[1:] == (543, 3), "Numpy file does not have correct dimensions"
        
        # Convert the numpy array to a torch tensor
        features = self.preprocess(npy_file)
        features = torch.from_numpy(features)

        # Pad or truncate label to be of self.max_wordlen
        label = torch.tensor([self.char_to_idx[x] for x in self.labels[idx]])
        worldlength = len(label)
        if worldlength >= self.max_wordlen:
            label = label[:self.max_wordlen]  # truncate if label is longer than self.max_wordlen
            worldlength = self.max_wordlen
        else:
            label = torch.cat([label,self.word_pad*torch.ones(self.max_wordlen-worldlength)])
        frames,points = features.shape        
        if frames >= self.max_framelen:
            features = features[:self.max_framelen]  # truncate if label is longer than self.max_wordlen
            frames = self.max_framelen
        else:
            features = torch.cat([features,self.frames_pad*torch.ones(self.max_framelen-frames,points)])

        return features.float(), label.long(),torch.tensor(frames).long(),torch.tensor(worldlength).long()
