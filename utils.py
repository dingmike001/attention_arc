from scipy.io import loadmat
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class SEED_VIG_load():
    def __init__(self, eeg_root, label_root, random_state=0, train_percentage=0.75, mode='train'):
        self.eeg_root = eeg_root
        self.label_root = label_root
        self.random_state = random_state
        self.train_percentage = train_percentage
        self.mode = mode
        self.total_eeg = np.transpose(loadmat(self.eeg_root)['re'], (0, 2, 1))
        self.total_labels = loadmat(self.label_root)['perclos_label']
        self.eeg, self.labels = self.train_val_split()

    def __len__(self):
        return len(self.labels)

    def train_val_split(self):
        self.train_data, self.val_data, self.train_label, self.val_label = train_test_split(self.total_eeg,
                                                                                            self.total_labels,
                                                                                            test_size=1 - self.train_percentage,
                                                                                            train_size=self.train_percentage,
                                                                                            random_state=self.random_state)

        if self.mode == 'train':
            return self.train_data, self.train_label
        return self.val_data, self.val_label

    def __getitem__(self, idx):
        eeg = self.eeg[idx]
        label = self.labels[idx]
        eeg_sample = {'eeg': eeg, 'label': label}
        return eeg_sample


def loaddata(testdata_path, labeldata_path):
    dataset_train = SEED_VIG_load(testdata_path, labeldata_path, random_state=0, train_percentage=0.75, mode='train')
    dataset_val = SEED_VIG_load(testdata_path, labeldata_path, random_state=0, train_percentage=0.75, mode='val')
    data_train = DataLoader(dataset_train, batch_size=16, shuffle=True)
    data_val = DataLoader(dataset_val, batch_size=1, shuffle=False)
    return data_train, data_val
