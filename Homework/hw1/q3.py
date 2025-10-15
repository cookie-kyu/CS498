# ###Q3: allreduce###
# ###please implement ring_allreduce method, using  pytorch's dist method is not allowed###

# from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
# import torch
# import torch.distributed as dist

# def reduce_scatter(chunks, tmp, world, rank, left, right):
#     #                                                                   #
#     #                                                                   #
#     # your code here: follow slides instruction: do counter-clockwise iteration
#     # for i in range(world - 1):
#     #     send_idx = (rank - i) % world
#     #     dest_idx = (rank - i - 1) % world

#     #     sbuf = chunks[send_idx].clone()

#     #     send_req = dist.isend(sbuf, dst=left)
#     #     dist.irecv(tmp, src=right)
#     #     #send_req.wait()
#     #     chunks[dest_idx] += tmp

#     # return chunks[rank]
#     for i in range(world - 1):
#         send_idx = (rank - i) % world
#         dest_idx = (rank - i - 1) % world

#         sbuf = chunks[send_idx].clone()

#         send_req = dist.isend(sbuf, dst=left)
#         recv_req = dist.irecv(tmp, src=right)
#         recv_req.wait()
#         send_req.wait()

#         chunks[dest_idx] += tmp
#     return chunks[rank]
#     #                                                                   #
#     #                                                                   #
#     #return
        
# def all_gather(chunks, tmp, current, world, rank, left, right):
#     #                                                                   #
#     #                                                                   #
#     # your code here: follow slides instruction: do counter-clockwise iteration
#     chunks[rank] = current.clone()

#     for i in range(world - 1):
#         send_idx = (rank + i) % world
#         recv_idx = (rank + i + 1) % world

#         sbuf = chunks[send_idx]

#         send_req = dist.isend(sbuf, dst=right)
#         dist.recv(tmp, src=left)
#         send_req.wait()

#         chunks[recv_idx] = tmp.clone()

#     return torch.cat(chunks, dim=0)
#     #                                                                   #
#     #                                                                   #
#     #return

# def ring_allreduce_(tensor: torch.Tensor, world_size = None, rankid = None):
#     """In-place ring all-reduce (SUM, optional average) using isend/irecv."""
#     world = world_size
#     if world == 1: return tensor
#     rank = rankid
#     left, right = (rank - 1) % world, (rank + 1) % world

#     ##following steps try to fill blank to the tensor so that final tensor can be divided to 3 chunks evenly
#     flat = tensor.contiguous().view(-1)
#     n = flat.numel()
#     chunk = (n + world - 1) // world
#     #                                                                   #
#     #                                                                   #
#     # your code here: we cannot divide flat into 3 pieces evenly as the
#     # flat lengh may not be able to divided exactly by 3....
    
#     #
#     #                                                                   #
#     #                                                                   #
#     #So, fill zeros at the end of flat to generate padded_flat
#     #padded_flat = None # modify this line and fill correct value into padded_flat
    
#     padded_flat = torch.zeros(chunk * world, dtype=flat.dtype, device=flat.device)
#     padded_flat[:n] = flat
#     chunks = [padded_flat[i*chunk:(i+1)*chunk] for i in range(world)]

#     #                                                                   #
#     #                                                                   #
#     # your code here: call reduce_scatter and all_gather
#     tmp = torch.zeros_like(chunks[0])
#     reduced_chunk = reduce_scatter(chunks, tmp, world, rank, left, right)
#     gathered_chunks = all_gather(chunks, tmp, reduced_chunk, world, rank, left, right)
#     gathered_chunks = gathered_chunks[:n].contiguous()

#     flat.copy_(torch.cat(gathered_chunks)[:n])
#     #
#     #                                                                   #
#     #                                                                   #
#     #we provide the reduce_scatter and all_gather func prototype for you
#     # You may adjust the function signature (input structure) of `reduce_scatter` and `all_gather` if needed.
    
#     # stitch & unpad  
#     flat /= world
#     tensor.view(-1).copy_(flat[:n])
#     return
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist

def reduce_scatter(chunks, tmp, world, rank, left, right):
    """Reduce-scatter phase (counter-clockwise communication)."""
    for i in range(world - 1):
        send_idx = (rank - i) % world
        recv_idx = (rank - i - 1) % world

        send_req = dist.isend(tensor=chunks[send_idx], dst=right)
        recv_req = dist.irecv(tensor=tmp, src=left)
        recv_req.wait()
        send_req.wait()

        chunks[recv_idx] += tmp 


def all_gather(chunks, tmp, world, rank, left, right):
    """All-gather phase (counter-clockwise communication)."""
    for i in range(world - 1):
        send_idx = (rank - i - 1) % world
        recv_idx = (rank - i - 2) % world

        send_req = dist.isend(tensor=chunks[send_idx], dst=right)
        recv_req = dist.irecv(tensor=tmp, src=left)
        recv_req.wait()
        send_req.wait()

        chunks[recv_idx].copy_(tmp)


def ring_allreduce_(tensor: torch.Tensor, world_size=None, rankid=None):
    """In-place ring all-reduce (sum and average) using point-to-point communication."""
    world = world_size
    if world == 1:
        return tensor
    rank = rankid
    left, right = (rank - 1) % world, (rank + 1) % world

    flat = tensor.contiguous().view(-1)
    n = flat.numel()
    chunk = (n + world - 1) // world

    
    padded_size = chunk * world
    padded_flat = torch.cat([flat, torch.zeros(padded_size - n, device=flat.device)])
    chunks = [padded_flat[i * chunk:(i + 1) * chunk] for i in range(world)]

    tmp = torch.zeros_like(chunks[0])

    
    reduce_scatter(chunks, tmp, world, rank, left, right)

    
    all_gather(chunks, tmp, world, rank, left, right)

    
    flat.copy_(torch.cat(chunks)[:n] / world)
    return tensor
