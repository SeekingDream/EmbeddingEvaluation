from torch.utils.data import Dataset
import pickle
import random


class CodeLoader(Dataset):
    def __init__(self, file_name, max_size):
        with open(file_name, 'rb') as f:
            dataset = pickle.load(f)
            self.dataset = dataset
            random.shuffle(self.dataset)

        if max_size is not None:
            self.dataset = self.dataset[:max_size]

    def __getitem__(self, index):
        data = self.dataset[index]
        code_context, target = data
        return code_context, target

    def __len__(self):
        return len(self.dataset)
