<img src="./images/计算与通信重叠.png" style="zoom:67%;" />

# MiniDDP

从零实现 PyTorch DistributedDataParallel (DDP) 的核心机制

## Key Features

* 从最小正确实现开始，逐步引入计算与通信重叠与 Grad Bucket 机制；
* 代码支持**多卡设备**和**纯 CPU** 运行，哪怕只用一台笔记本，也可以运行代码，足够理解 MiniDDP 实现逻辑；

## Quick Start

```bash
git clone https://github.com/Pearblossom-M/MiniDDP-from-Scratch.git
cd MiniDDP-from-Scratch/src/MiniDDP-with-bucket
torchrun --standalone --nproc_per_node=2 train.py
```

如果是纯 CPU 运行，需要将 `src/MiniDDP-with-bucket/train.py` 的 `train` 函数入参设置为：`use_cpu = True`。

* `src/MiniDDP-naive`

  MiniDDP 的最小正确实现；

* `src/MiniDDP-with-hook`

  使用 Autograd Hook 机制，引入计算与通信重叠，加快训练速度；

* `src/MiniDDP-with-bucket`

  增加 Grad Bucket 机制，减少通信启动次数，提高带宽利用率；

## Tutorial

详细教程位于：`从零实现PyTorch_DDP核心机制.md`