###Q3: allreduce###
###please implement ring_allreduce method, using  pytorch's dist method is not allowed###

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist

def reduce_scatter(chunks, tmp, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    for i in range(world - 1):
        send_idx = (rank - i) % world
        recv_idx = (rank - i - 1) % world

        send_req = dist.isend(tensor=chunks[send_idx], dst=right)
        recv_req = dist.irecv(tensor=tmp, src=left)
        recv_req.wait()
        send_req.wait()

        chunks[recv_idx] += tmp 
    #                                                                   #
    #                                                                   #
    return


def all_gather(chunks, tmp, world, rank, left, right):
    # your code here: follow slides instruction: do counter-clockwise iteration
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
    if world == 1: return tensor
    rank = rankid
    left, right = (rank - 1) % world, (rank + 1) % world

    ##following steps try to fill blank to the tensor so that final tensor can be divided to 3 chunks evenly
    flat = tensor.contiguous().view(-1)
    n = flat.numel()
    chunk = (n + world - 1) // world
    #                                                                   #
    #                                                                   #
    # your code here: we cannot divide flat into 3 pieces evenly as the
    # flat lengh may not be able to divided exactly by 3....
    padded_size = chunk * world
    #
    #                                                                   #
    #                                                                   #
    #So, fill zeros at the end of flat to generate padded_flat
    padded_flat = torch.cat([flat, torch.zeros(padded_size - n, device=flat.device)])
    chunks = [padded_flat[i * chunk:(i + 1) * chunk] for i in range(world)]
    #                                                                   #
    #                                                                   #
    # your code here: call reduce_scatter and all_gather
    tmp = torch.zeros_like(chunks[0])
    reduce_scatter(chunks, tmp, world, rank, left, right)
    all_gather(chunks, tmp, world, rank, left, right)
    #
    #                                                                   #
    #                                                                   #
    #we provide the reduce_scatter and all_gather func prototype for you
    # You may adjust the function signature (input structure) of `reduce_scatter` and `all_gather` if needed.


    # stitch & unpad 
    flat.copy_(torch.cat(chunks)[:n] / world) # _ 
    return tensor
