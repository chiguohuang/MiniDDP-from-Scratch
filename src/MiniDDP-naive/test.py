import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from MiniDDP import MiniDDP

# congfig
class config:
    BATCH_SIZE = 32       # 适当增大, 让计算时间相对噪声更显著
    IN_DIM = 256          # 同上
    HIDDEN_DIM = 1024     # 同上
    OUT_DIM = 1
    NUM_LAYERS = 20       # MLP 层数
    WARMUP_STEPS = 20     # warmup 步数
    MEASURE_STEPS = 200   # 正式计时步数

class DeepMLP(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=256, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

        layers = []
        for _ in range(config.NUM_LAYERS):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc2(self.net(self.relu(self.fc1(x))))

def run_step(ddp_model, optimizer, criterion, device):
    x = torch.randn(config.BATCH_SIZE, config.IN_DIM, device=device)
    y = torch.randn(config.BATCH_SIZE, config.OUT_DIM, device=device)
    optimizer.zero_grad()
    output = ddp_model(x)
    loss = criterion(output, y)
    loss.backward()
    ddp_model.all_reduce_gradients()
    optimizer.step()

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
    model = DeepMLP(
        in_dim=config.IN_DIM,
        hidden_dim=config.HIDDEN_DIM, 
        out_dim=config.OUT_DIM
    ).to(device)    
    ddp_model = MiniDDP(model)

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # warmup
    for _ in range(config.WARMUP_STEPS):
        run_step(ddp_model, optimizer, criterion, device)
    dist.barrier()   # 确保所有进程都完成 warmup 再开始计时

    if device.type == "cuda":
        torch.cuda.synchronize()

    # 正式计时
    start = time.perf_counter()
    for _ in range(config.MEASURE_STEPS):
        run_step(ddp_model, optimizer, criterion, device)
    dist.barrier()

    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000
    if dist.get_rank() == 0:
        print(f"{elapsed_ms:.3f} ms")
        print(f"{elapsed_ms/config.MEASURE_STEPS:.3f} ms/iter")

    # clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    train(True)

# torchrun --standalone --nproc_per_node=2 test.py