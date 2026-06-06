import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Tuple, Dict

# ===================================================================
#  GradBucket: 负责单个桶的打包 / all-reduce / 解包                            #
# =================================================================== 
class GradBucket:
    """
    核心思路：将桶内若干参数的梯度预先拷贝进一块连续 flat_buffer,
    对这块 buffer 发起一次 all-reduce, 再把结果写回各 param.grad, 
    这样 K 个桶只需 K 次通信, 而非原来的 N_params 次。
    """
    def __init__(self, params: List[nn.Parameter], bucket_id: int) -> None:
        self.params = params
        self.bucket_id = bucket_id
 
        device = params[0].device
        dtype  = params[0].dtype  # 假定同桶参数 dtype 一致
 
        # 预分配连续 flat_buffer (all-reduce 的唯一操作对象)
        total_numel = sum(p.numel() for p in params)
        self.flat_buffer = torch.zeros(total_numel, dtype=dtype, device=device)
 
        # 为每个参数在 flat_buffer 里开一个视图 (零拷贝)
        # 视图和 flat_buffer 共享底层存储, 写视图 = 写 flat_buffer
        self.param_views: List[torch.Tensor] = []
        offset = 0
        for p in params:
            n = p.numel()
            self.param_views.append(self.flat_buffer[offset: offset + n])
            offset += n
 
        self.handle = None           # 异步 all-reduce 句柄
        self._pending = len(params)  # 还有几个梯度尚未就绪
 
    def mark_ready(self, param_idx: int) -> bool:
        """
        某个参数的梯度已写入 param.grad, 将其复制进 flat_buffer, 
        返回 True 表示本桶所有梯度均已就绪, 可以发起 all-reduce。
        """
        param = self.params[param_idx]
        if param.grad is not None:
            # 把 param.grad 写入 flat_buffer
            # view(-1) 保证形状为 1D, 与 param_views 的切片一致
            self.param_views[param_idx].copy_(param.grad.view(-1))
 
        self._pending -= 1
        return self._pending == 0  # 降到 0 → 桶满 → 可以发起 all-reduce
 
    def launch_allreduce(self, world_size: int) -> None:
        """
        桶满后立即异步启动 all-reduce
        """
        self._world_size = world_size
        self.handle = dist.all_reduce(
            self.flat_buffer, op=dist.ReduceOp.SUM, async_op=True
        )
 
    def finalize(self) -> None:
        """
        阻塞等待 all-reduce 完成, 平均后将结果写回各 param.grad, 
        在 optimizer.step() 之前、MiniDDP.wait_for_gradients() 内调用。
        """
        self.handle.wait()
        self.flat_buffer.div_(self._world_size)
 
        # 将 flat_buffer 的各段写回对应的 param.grad
        for param, view in zip(self.params, self.param_views):
            if param.grad is not None:
                param.grad.copy_(view.view_as(param.grad))
 
    def reset(self) -> None:
        """
        迭代结束后重置, 供下一次 backward 复用 (避免重新分配)
        """
        self._pending = len(self.params)
        self.handle = None

# ===================================================================
#  MiniDDP (带 bucket)
# =================================================================== 
class MiniDDP(nn.Module):
    """
    Minimal DDP wrapper with bucket.

    相比逐参数 all-reduce 的改进：
    - 通信次数: N_params → N_buckets (通常 N_buckets << N_params)
    - 每次 all-reduce 操作的数据块更大, 带宽利用率更高
    """
    def __init__(self, module: nn.Module, bucket_size_mb: float = 25.0) -> None:
        super().__init__()
        self.module     = module
        self.world_size = dist.get_world_size()
        self.rank       = dist.get_rank()
        self._sync_module_states()
 
        # param id → (所属桶, 在桶内的下标)
        self._param_bucket_map: Dict[int, Tuple[GradBucket, int]] = {}
        self.buckets: List[GradBucket] = []
        self._build_buckets(bucket_size_mb)
 
        # 记录本次 backward 中已发起 all-reduce 的桶
        self._active_buckets: List[GradBucket] = []
 
        self._register_gradient_hooks()
    
    def _sync_module_states(self) -> None:
        module_states = list(self.module.parameters()) + list(self.module.buffers())
        with torch.no_grad():
            for tensor in module_states:
                dist.broadcast(tensor, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def _build_buckets(self, bucket_size_mb: float) -> None:
        """
        按反向传播顺序为参数分配桶, 每个桶的字节上限为 bucket_size_mb MB
        """
        bucket_bytes = int(bucket_size_mb * 1024 * 1024)
 
        # 收集需要梯度的参数，翻转为反向传播顺序
        trainable = [p for p in self.module.parameters() if p.requires_grad]
        trainable = list(reversed(trainable))
 
        cur_params: List[nn.Parameter] = []
        cur_bytes = 0
 
        for param in trainable:
            param_bytes = param.numel() * param.element_size()
            # 当前桶非空且再加入此参数会超限 → 先封桶
            if cur_params and cur_bytes + param_bytes > bucket_bytes:
                self._seal_bucket(cur_params)
                cur_params, cur_bytes = [], 0
            cur_params.append(param)
            cur_bytes += param_bytes
 		# 处理最后一个桶
        if cur_params:
            self._seal_bucket(cur_params)

        if self.rank == 0:
            sizes = [sum(p.numel() * 4 for p in b.params) / 1024 / 1024
                     for b in self.buckets]
            print(f"[MiniDDP] {len(self.buckets)} bucket(s): "
                  f"{[f'{s:.3f}MB' for s in sizes]}")
 
    def _seal_bucket(self, params: List[nn.Parameter]) -> None:
        """
        封装一个桶并建立 param → (bucket, idx) 的索引
        """
        bucket = GradBucket(params, bucket_id=len(self.buckets))
        for idx, p in enumerate(params):
            self._param_bucket_map[id(p)] = (bucket, idx)
        self.buckets.append(bucket)
    
    def _register_gradient_hooks(self) -> None:
        """
        为所有需要计算梯度的参数注册反向传播 Hook
        """
        for param in self.module.parameters():
            if param.requires_grad:              
                # 触发时机：param.grad 已经写入完毕
                param.register_post_accumulate_grad_hook(self._make_hook())
    
    def _make_hook(self):
        """
        工厂函数, 返回一个被 Autograd 调用的 hook 函数, 
        hook 接收的 param 参数即触发 hook 的那个参数本体 (autograd 传入), 
        通过 id(param) 查到它所属的桶, 标记就绪并在桶满时发起 all-reduce。
        """ 
        def hook(param: nn.Parameter) -> None:
            bucket, param_idx = self._param_bucket_map[id(param)]
            bucket_full = bucket.mark_ready(param_idx)
            if bucket_full:
                bucket.launch_allreduce(self.world_size)
                self._active_buckets.append(bucket)
        return hook

    def wait_for_gradients(self) -> None:
        """
        在 optimizer.step() 之前调用：
        1. 按桶的发起顺序逐个 wait, 并将结果写回 param.grad
        2. 重置所有桶供下一次 backward 复用
        """
        for bucket in self._active_buckets:
            bucket.finalize()
 
        for bucket in self.buckets:
            bucket.reset()
 
        self._active_buckets.clear()