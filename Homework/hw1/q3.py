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
    """
    Ring Reduce-Scatter phase:
    Each rank reduces chunks by passing them counter-clockwise.
    """
    for i in range(world - 1):
        # which chunk to send
        send_chunk_idx = (rank - i) % world
        recv_chunk_idx = (rank - i - 1) % world

        send_req = dist.isend(chunks[send_chunk_idx], dst=left)
        recv_req = dist.irecv(tmp, src=right)
        recv_req.wait()
        send_req.wait()

        # accumulate received data into the correct chunk
        chunks[recv_chunk_idx] += tmp


def all_gather(chunks, tmp, current, world, rank, left, right):
    """
    Ring All-Gather phase:
    After reduce-scatter, each rank has one reduced chunk. Circulate it
    counter-clockwise so everyone gets all reduced chunks.
    """
    for i in range(world - 1):
        send_chunk_idx = (rank - i) % world
        recv_chunk_idx = (rank - i - 1) % world

        send_req = dist.isend(chunks[send_chunk_idx], dst=left)
        recv_req = dist.irecv(tmp, src=right)
        recv_req.wait()
        send_req.wait()

        # store received chunk in the right place
        chunks[recv_chunk_idx].copy_(tmp)


def ring_allreduce_(tensor: torch.Tensor, world_size=None, rankid=None):
    """
    In-place ring all-reduce using only send/recv (no dist.all_reduce).
    """
    world = world_size
    if world == 1:
        return tensor
    rank = rankid
    left, right = (rank - 1) % world, (rank + 1) % world

    # Flatten and pad to make divisible by world size
    flat = tensor.contiguous().view(-1)
    n = flat.numel()
    chunk = (n + world - 1) // world  # ceil division

    pad_len = chunk * world - n
    padded_flat = torch.cat([flat, torch.zeros(pad_len, dtype=flat.dtype, device=flat.device)])

    chunks = [padded_flat[i*chunk:(i+1)*chunk] for i in range(world)]
    tmp = torch.empty_like(chunks[0])

    # Reduce-Scatter phase
    reduce_scatter(chunks, tmp, world, rank, left, right)

    # All-Gather phase
    all_gather(chunks, tmp, chunks[rank], world, rank, left, right)

    # Reconstruct final tensor
    flat = torch.cat(chunks)
    flat /= world
    tensor.view(-1).copy_(flat[:n])
    return tensor