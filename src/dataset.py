from torch.utils.data import Dataset


class TrigramDataset(Dataset):
    def __init__(self, ids, context_len=3):
        if ids.dim() != 1:
            raise ValueError("ids must be a 1D tensor")
        if len(ids) <= context_len:
            raise ValueError("ids length must be greater than context_len")
        self.ids = ids
        self.context_len = context_len

    def __len__(self):
        return len(self.ids) - self.context_len

    def __getitem__(self, idx):
        x = self.ids[idx:idx + self.context_len]
        y = self.ids[idx + self.context_len]
        return x, y
