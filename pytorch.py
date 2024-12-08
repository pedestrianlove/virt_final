import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Dummy dataset
class DummyDataset(Dataset):
    def __init__(self, size=10000, features=100):
        self.data = torch.randn(size, features)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# Simple model
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# Training function
def train(rank, world_size, backend='nccl'):
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Create model and move it to GPU
    model = SimpleModel(input_size=100, output_size=2).to(device)
    model = DDP(model, device_ids=[rank])

    # Create dataset and dataloader
    dataset = DummyDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    start_time = time.time()
    for epoch in range(5):  # 5 epochs
        sampler.set_epoch(epoch)  # Shuffle dataset
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    if rank == 0:  # Log only on rank 0
        print(f"Training completed in {time.time() - start_time:.2f} seconds on {world_size} GPUs")

    dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    from torch.multiprocessing import spawn

    # Set default MASTER_ADDR and MASTER_PORT if not set
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=4, help='Number of GPUs to use')
    args = parser.parse_args()

    world_size = args.gpus

    if world_size == 1:
        # Single GPU training
        print("Running on a single GPU")
        start_time = time.time()
        single_gpu_model = SimpleModel(input_size=100, output_size=2).cuda()
        single_gpu_dataset = DummyDataset()
        single_gpu_loader = DataLoader(single_gpu_dataset, batch_size=64)
        single_gpu_optimizer = optim.SGD(single_gpu_model.parameters(), lr=0.01)
        single_gpu_criterion = nn.CrossEntropyLoss()

        for epoch in range(5):
            for inputs, targets in single_gpu_loader:
                inputs, targets = inputs.cuda(), targets.cuda()

                single_gpu_optimizer.zero_grad()
                outputs = single_gpu_model(inputs)
                loss = single_gpu_criterion(outputs, targets)
                loss.backward()
                single_gpu_optimizer.step()

        print(f"Single GPU training completed in {time.time() - start_time:.2f} seconds")

    else:
        # Multi-GPU DDP training
        print("\nRunning with DDP on multiple GPUs")
        spawn(train, args=(world_size,), nprocs=world_size)
