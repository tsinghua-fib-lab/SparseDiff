import numpy as np
import torch
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader, random_split


def reconstruct_from_sparse(Sparse, encoding_indices, n, grid_size=128, strategy='first'):   
    C, total_M = Sparse.shape
    H = W = grid_size

    assert encoding_indices.shape == (C, H * W)
    Sparse_grouped = Sparse.view(C, -1, n)

    if strategy == 'first':
        Sparse_selected = Sparse_grouped[:, :, 0]  # shape: (C, M)
    elif strategy == 'mean':
        Sparse_selected = Sparse_grouped.mean(dim=2)  # shape: (C, M)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    
    index_map = encoding_indices  # shape: (C, H*W)

    reconstructed_flat = torch.gather(Sparse_selected, dim=1, index=index_map)  # shape: (C, H*W)
    reconstructed = reconstructed_flat.view(C, H, W)
    return reconstructed

def generate_sample_sparse_observations(encoding_indices, data, n, M=100):
    C, HW = encoding_indices.shape
    H = W = int(HW ** 0.5)
    data = data.reshape(C, HW)
    total_M = M * n
    Sparse = torch.zeros((C, total_M))

    for c in range(C):
        indices = encoding_indices[c]  # shape: (H*W,)
        unique_codes = torch.unique(indices)

        for code in unique_codes:
            code_int = int(code.item())

            mask = (indices == code)
            selected_idx = torch.nonzero(mask).squeeze()
            if selected_idx.dim() == 0:
                selected_idx = selected_idx.unsqueeze(0)

            num_available = selected_idx.shape[0]

            target_start = code_int * n

            if num_available < n:

                for i in range(num_available):
                    idx = selected_idx[i]
                    x, y = divmod(idx.item(), W)
                    
                    Sparse[c, target_start + i] = data[c, idx]
                if num_available > 0:
                    repeat_needed = n - num_available
                    repeated = selected_idx[torch.randint(0, num_available, (repeat_needed,))]
                    for i, idx in enumerate(repeated):
                        x, y = divmod(idx.item(), W)
                       
                        Sparse[c, target_start + num_available + i] = data[c, idx]
            else:
                chosen_indices = selected_idx[torch.randperm(num_available)[:n]]
                for i, idx in enumerate(chosen_indices):
                   
                    Sparse[c, target_start + i] = data[c, idx]

    return Sparse

class SystemDataset_cond(Dataset):
    def __init__(self, path, n, grid_size=128, strategy='mean'):
        self.path = path
        self.data = self._load_data()  # shape: (N, T, C, H, W)
        self.n = n
        self.grid_size = grid_size
        self.strategy = strategy
        
        # Precompute the sparse observations and reconstructions for all data
       
        self.reconstructed_data = []
        
        for data_point in self.data:
            x, encoding_indices = data_point.x, data_point.y  # shape: (C, H, W), (C, H*W)
            Sparse = generate_sample_sparse_observations(encoding_indices, x, self.n, M=100)
            reconstructed = reconstruct_from_sparse(Sparse, encoding_indices, self.n, self.grid_size, self.strategy)
            self.reconstructed_data.append(reconstructed)

    def _load_data(self):
        uv = torch.load(self.path)  # shape: (N_traj, T, C, H, W)
        return uv

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Access the precomputed sparse and reconstructed data
        x = self.data[idx].x  # shape: (C, H, W), (C, H*W)
        reconstructed = self.reconstructed_data[idx]
        return x, reconstructed

#
class SystemDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = self._load_data() # shape: (N, T, C, H, W)

    def _load_data(self):
        uv = np.load(self.path) # shape: (N_traj, T, C, H, W)
        _, _, C, H, W = uv.shape
        uv = uv.reshape(-1, C, H, W)
        uv = torch.tensor(uv, dtype=torch.float32) # shape: (N*T*ratio, C, H, W)
        # Normalize to [0, 1]
        self.min = uv.amin(dim=(0, 2, 3), keepdim=True)
        self.max = uv.amax(dim=(0, 2, 3), keepdim=True)
        uv = (uv - self.min) / (self.max - self.min)
        
        return uv

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        uv = self.data[idx] # shape: (C, H, W)    
        return uv

class SystemDataset_gled(Dataset):
    def __init__(self, path, horizon):
        self.path = path
        self.horizon = horizon
        self.data = self._load_data() # shape: (N, T, C, H, W)

    def _load_data(self):
        uv = np.load(self.path) # shape: (N_traj, T, 2, H, W)
        n_traj, T, C, H, W = uv.shape
        
        # Split each trajectory into multiple sequences
        seq_num_per_traj = T // (self.horizon + 1)
        uv = uv[:, :seq_num_per_traj * (self.horizon + 1)].reshape(n_traj, seq_num_per_traj, self.horizon + 1, C, H, W)
        uv = uv.reshape(-1, self.horizon + 1, C, H, W)
        
        uv = torch.tensor(uv, dtype=torch.float32)
        
        # Normalize to [0, 1]
        self.min = uv.amin(dim=(0, 1, 3, 4), keepdim=True)
        self.max = uv.amax(dim=(0, 1, 3, 4), keepdim=True)
        uv = (uv - self.min) / (self.max - self.min)
        
        return uv

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        uv = self.data[idx] # shape: (T, 2, H, W)    
        return uv

class SystemDataset_gled_1(Dataset):
    def __init__(self, path, lookback, horizon):
        self.path = path
        self.horizon = horizon
        self.lookback = lookback
        self.data = self._load_data() # shape: (N, T, C, H, W)

    def _load_data(self):
        uv = np.load(self.path) # shape: (N_traj, T, 2, H, W)
        n_traj, T, C, H, W = uv.shape
        
        # Split each trajectory into multiple sequences
        seq_num_per_traj = T // (self.horizon + self.lookback)
        uv = uv[:, :seq_num_per_traj * (self.horizon + self.lookback)].reshape(n_traj, seq_num_per_traj, self.horizon + self.lookback, C, H, W)
        uv = uv.reshape(-1, self.horizon + self.lookback, C, H, W)
        
        uv = torch.tensor(uv, dtype=torch.float32)
        
        # Normalize to [0, 1]
        self.min = uv.amin(dim=(0, 1, 3, 4), keepdim=True)
        self.max = uv.amax(dim=(0, 1, 3, 4), keepdim=True)
        uv = (uv - self.min) / (self.max - self.min)
        
        return uv

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        uv = self.data[idx] # shape: (T, 2, H, W)    
        return uv


def get_dataset(name, grid_size = 128, strategy='mean'):
    if name == 'bruss':
        path = 'data/bru/uv.npy'
    elif name == 'lo_diffusion':
        path = 'data/lo/cond_diffusion.pth'
    elif name == 'sh':
        path = '/data5/chengjingwen/sh/uv.npy'
    elif name == 'ns':
        path = '/data5/chengjingwen/ns/uv.npy'
    elif name == 'cy':
        path = '/data5/chengjingwen/cy/uv.npy'
    elif name == 'sevir_tem':
        path = '/data5/chengjingwen/sevir_tem/uv.npy'
    else:
        raise ValueError(f"Invalid dataset name: {name}")
    
    DATASET = SystemDataset
    
    return DATASET(path=path)

def get_dataset_gled(name, lookback, horizon):
    if name == 'sevir_tem':
        path = '/data5/chengjingwen/sevir_tem/uv.npy'
    elif name == 'cy':
        path = '/data5/chengjingwen/cy/uv.npy'
    elif name == 'lo':
        path = '/data5/chengjingwen/lo/uv.npy'
    elif name == 'sh':
        path = '/data5/chengjingwen/sh/uv.npy'
    elif name == 'ns':
        path = '/data5/chengjingwen/ns/uv.npy'
    else:
        raise ValueError(f"Invalid dataset name: {name}")
    
    DATASET = SystemDataset_gled_1
    RES = 64
    
    return DATASET(path=path, lookback= lookback, horizon=horizon)
