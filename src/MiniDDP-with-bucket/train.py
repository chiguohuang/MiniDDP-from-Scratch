import os
import torch
import torch.nn as nn
import torch.distributed as dist
from MiniDDP import MiniDDP

# congfig
class config:
    BATCH_SIZE = 1
    IN_DIM = 16
    STEP = 1

class ToyModel(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def train(use_cpu: bool = False):
    # 1. Process-group initialization
    if use_cpu:
        dist.init_process_group(backend="gloo",)
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cpu")
    else:
        dist.init_process_group(backend="nccl",)
        local_rank = int(os.environ["LOCAL_RANK"])
    
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    # 2. Wrap model with MiniDDP
    model = ToyModel(in_dim=config.IN_DIM).to(device)
    print(
        f"[Rank {local_rank}] before MiniDDP "
        f"fc1.weight[0,0] = {model.fc1.weight[0,0].item():.6f}"
    )

    ddp_model = MiniDDP(model)

    print(
        f"[Rank {local_rank}] after MiniDDP "
        f"fc1.weight[0,0] = {model.fc1.weight[0,0].item():.6f}"
    )
    
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 3. Training loop
    for _ in range(config.STEP):
        # generate input
        x = torch.randn(config.BATCH_SIZE, config.IN_DIM, device=device)
        y = torch.randn(config.BATCH_SIZE, 1, device=device)

        # forward
        optimizer.zero_grad()
        output = ddp_model(x)
        loss = criterion(output, y)

        # backward
        # 此时 hook 会被挨个触发，异步通信在后台启动       
        loss.backward()

        # all-reduce gradients
        ddp_model.wait_for_gradients()        
        print(
            f"[Rank {local_rank}] after All Reduce "
            f"grad = {model.fc1.weight.grad[0,0].item():.6f}"
        )

        # optimize
        optimizer.step()
        print(
            f"[Rank {local_rank}] after optimize "
            f"fc1.weight[0,0] = {model.fc1.weight[0,0].item():.6f}"
        )
    
    # clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    train(True)

# torchrun --standalone --nproc_per_node=2 train.py