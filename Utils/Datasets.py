import torch
from  torch.utils.data import Dataset

class FNN_dataset(Dataset):
    def __init__(self,chromossomes,fitness):
        self.X = torch.tensor(chromossomes,dtype=torch.float)
        self.Y = torch.tensor(fitness,dtype=torch.float)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx],self.Y[idx]
        