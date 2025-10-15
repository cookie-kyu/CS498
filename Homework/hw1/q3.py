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
import torch
import torch.distributed as dist

def reduce_scatter(chunks, tmp, world, rank, left, right, debug=False):
    """
    Counter-clockwise reduce-scatter (defensive):
    On step s (0..world-2) receive chunk (rank - s - 1) from right,
    add it into our local slot for that chunk index, then send chunk (rank - s)
    to left (but clone the send buffer to avoid aliasing).
    """
    for s in range(world - 1):
        send_idx = (rank - s) % world
        recv_idx = (rank - s - 1) % world

        # post irecv into tmp and wait
        recv_req = dist.irecv(tensor=tmp, src=right)
        recv_req.wait()
        if debug:
            print(f"[rank {rank}] reduce_scatter step {s} recv into tmp for idx {recv_idx}")

        # accumulate received data into the slot recv_idx
        chunks[recv_idx].add_(tmp)

        # prepare send buffer (clone to avoid modification while sending)
        send_buf = chunks[send_idx].clone()
        send_req = dist.isend(tensor=send_buf, dst=left)
        send_req.wait()
        if debug:
            print(f"[rank {rank}] reduce_scatter step {s} sent cloned chunk {send_idx} to {left}")


def all_gather(chunks, tmp, world, rank, left, right, debug=False):
    """
    Counter-clockwise all-gather (defensive):
    On step s (0..world-2) receive chunk (rank - s - 1) from right,
    place it into chunks[recv_idx], then send chunk (rank - s) to left (send a clone).
    """
    for s in range(world - 1):
        send_idx = (rank - s) % world
        recv_idx = (rank - s - 1) % world

        recv_req = dist.irecv(tensor=tmp, src=right)
        recv_req.wait()
        if debug:
            print(f"[rank {rank}] all_gather step {s} recv into tmp for idx {recv_idx}")

        # place received data
        chunks[recv_idx].copy_(tmp)

        # send cloned buffer
        send_buf = chunks[send_idx].clone()
        send_req = dist.isend(tensor=send_buf, dst=left)
        send_req.wait()
        if debug:
            print(f"[rank {rank}] all_gather step {s} sent cloned chunk {send_idx} to {left}")


def ring_allreduce_(tensor: torch.Tensor, world_size=None, rankid=None, debug=False):
    """
    In-place ring all-reduce (sum then average). Defensive with clones.
    """
    world = world_size
    if world == 1:
        return tensor
    rank = rankid
    left, right = (rank - 1) % world, (rank + 1) % world

    flat = tensor.contiguous().view(-1)
    n = flat.numel()
    chunk = (n + world - 1) // world  # ceil division
    pad_len = chunk * world - n

    if pad_len > 0:
        padded_flat = torch.cat([flat, torch.zeros(pad_len, dtype=flat.dtype, device=flat.device)])
    else:
        padded_flat = flat

    # Create independent chunk tensors (cloned) so we can safely modify them
    chunks = [padded_flat[i * chunk:(i + 1) * chunk].clone() for i in range(world)]
    tmp = torch.empty_like(chunks[0])

    # Optional debug: ensure shapes consistent across ranks
    if debug and rank == 0:
        print(f"[rank {rank}] n={n}, chunk={chunk}, pad_len={pad_len}, world={world}")

    # REDUCE-SCATTER
    reduce_scatter(chunks, tmp, world, rank, left, right, debug=debug)

    # After reduce-scatter, chunk index == rank contains the sum across ranks for that chunk.
    # ALL-GATHER
    all_gather(chunks, tmp, world, rank, left, right, debug=debug)

    # stitch back and unpad
    result = torch.cat(chunks)
    result /= float(world)   # average; remove if you only want SUM
    tensor.view(-1).copy_(result[:n])
    return tensor
