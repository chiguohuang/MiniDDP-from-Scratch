import torch
import torch.nn as nn
import torch.distributed as dist

class MiniDDP(nn.Module):
    """
    Minimal DDP wrapper
    """
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        # Return the number of processes in the current process group
        self.world_size = dist.get_world_size() 
        # Return the rank of current process within process group
        self.rank = dist.get_rank()
        # Parameter Synchronization
        self._sync_module_states()
    
    # =================================================================
    # Step 1: Parameter Synchronization
    # =================================================================
    def _sync_module_states(self) -> None:
        """
        Broadcast every parameter and buffer from rank 0 → all ranks.

        Collective : dist.broadcast  (one call per tensor)
        Postcondition : every rank holds byte-identical weight tensors.
        """
        module_states = list(self.module.parameters()) + list(self.module.buffers())
        with torch.no_grad():
            for tensor in module_states:
                # All ranks call broadcast; rank 0 is the source (src=0).
                # Ranks 1..N-1 receive rank-0's data, overwriting their own.
                dist.broadcast(tensor, src=0)

    # =================================================================
    # Step 2: Forward
    # =================================================================
    def forward(self, *args, **kwargs):
        """
        pass-through, no collective.
        """
        return self.module(*args, **kwargs)
    
    # =================================================================
    # Step 3: loss.backward(), called by the user between steps 2 and 4
    # =================================================================

    # =================================================================
    # Step 4: All-reduce gradients
    # =================================================================
    def all_reduce_gradients(self) -> None:
        """
        Average gradients across all ranks.
        """
        with torch.no_grad():
            for param in self.module.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad.div_(self.world_size)