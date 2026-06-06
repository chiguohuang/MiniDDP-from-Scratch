import torch
import torch.nn as nn
import torch.distributed as dist

class MiniDDP(nn.Module):
    """
    Minimal DDP wrapper with overlapped computation and communication
    """
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size() 
        self.rank = dist.get_rank()
        self._sync_module_states()

        # Collect handles of asynchronous communication
        self.reduction_handles = []
        # Register autograd hook
        self._register_gradient_hooks()
    
    def _sync_module_states(self) -> None:
        module_states = list(self.module.parameters()) + list(self.module.buffers())
        with torch.no_grad():
            for tensor in module_states:
                dist.broadcast(tensor, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
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
        工厂函数，返回一个被 Autograd 调用的 hook 函数
        """
        def hook(param): # ← 参数是 param, 不是 grad
            # 当这个 hook 被触发时, 立即发起异步的 all-reduce
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)           
            self.reduction_handles.append((param.grad, handle))
        return hook

    def wait_for_gradients(self) -> None:
        """
        在 optimizer.step() 之前调用,
        阻塞等待所有后台的异步通信完成, 并除以 world_size.
        """
        with torch.no_grad():
            # 遍历所有收集到的异步句柄
            for grad, handle in self.reduction_handles:
                handle.wait()  # 阻塞当前流, 直到这个 grad 的 all-reduce 完成
                # 这里的 grad 就是 param.grad, 原地修改即生效
                grad.div_(self.world_size) 
                
        # 清空列表, 为下一次 backward 做准备
        self.reduction_handles.clear()
    
    # def _register_gradient_hooks(self) -> None:
    #     """
    #     为所有需要计算梯度的参数注册反向传播 Hook
    #     """
    #     for param in self.module.parameters():
    #         if param.requires_grad:              
    #             param.register_hook(self._make_hook(param))
    
    # def _make_hook(self, param):
    #     """
    #     工厂函数，返回一个被 Autograd 调用的 hook 函数
    #     """
    #     def hook(grad):
    #         # 当这个 hook 被触发时, 立即发起异步的 all-reduce
    #         handle = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)           
    #         self.reduction_handles.append((param, grad, handle))
    #         # print(
    #         #     f"[Rank {self.rank}] "
    #         #     f"hook grad ptr = {grad.data_ptr()}"
    #         # )
    #         return grad 
    #     return hook

    # def wait_for_gradients(self) -> None:
    #     """
    #     在 optimizer.step() 之前调用,
    #     阻塞等待所有后台的异步通信完成，并除以 world_size.
    #     """
    #     with torch.no_grad():
    #         # 遍历所有收集到的异步句柄
    #         for param, grad, handle in self.reduction_handles:
    #             handle.wait()  # 阻塞当前流, 直到这个 grad 的 all-reduce 完成
    #             # print(
    #             #     f"[Rank {self.rank}] "
    #             #     f"grad ptr = {grad.data_ptr()}"
    #             # )
    #             # print(
    #             #     f"[Rank {self.rank}] "
    #             #     f"param.grad ptr = {param.grad.data_ptr()}"
    #             # )

    #             grad.div_(self.world_size)
    #             param.grad.copy_(grad)
                
    #     # 清空列表, 为下一次 backward 做准备
    #     self.reduction_handles.clear()