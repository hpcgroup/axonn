# Copyright 2021 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from . import config
from typing import Optional, List
from .communication import communication_handle
import torch
from mpi4py import MPI
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from enum import Enum
import apex.amp as amp


# True when init has been called
is_initialized = False
# Communication handle for point-to-point (MPI) and collective (NCCL) communication
comm_handle = None
# store references to input activation
input_tensors_cache = {}
# store references to output activation
output_tensors_cache = {}
# store (future object, tensor reference) for pending isends
transit_tensors = []
# store (future object, tensor reference) for pending irecvs
requests = {
    "fw": None,
    "bw": None,
}
# store reference to model shard
model = None
# loss function
criterion = None
# reference to flattened model params
model_params_fp32, model_params_fp16 = None, None
# reference to flattened model gradients
model_grads_fp32, model_grads_fp16 = None, None
# the computation dtype (one of fp16/fp32)
computation_dtype = None
# fp16 all reduce, only applicable with mixed precision
_fp16_all_reduce = None
# reference to optimizer
fp32_optimizer = None

class Operation(Enum):
    FW = 0
    BW = 1


class empty_dataset(torch.utils.data.Dataset):
    """
    Proxy dataset object for GPUs with inter_layer_parallel_rank > 0
    """

    def __init__(self, length: int, num_tensors: int):
        """Constructor for the proxy dataset class

        Arguments:
            length (int): number of datapoints in the dataset
            num_tensors (int): number of tensors per datapoint

        Returns:
            A PyTorch dataset object

        """
        self.length = length
        self.num_tensors = num_tensors

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return [0 for _ in range(self.num_tensors)]


def init(G_inter: int, G_data: int, gpus_per_node: Optional[int] = None, mixed_precision=False, fp16_allreduce=True) -> None:
    """
    Initialize AxoNN's 2D parallelism with G_inter-way inter-layer
    parallelism and G_data-way data parallelism

    Arguments:
        G_inter (int): number of GPUs used for inter-layer parallelism
        G_data (int): number of GPUs used for data parallelism
        gpus_per_node (int, optional):  number of GPUs per node, if not
            provided this is inferred using pytorch
        mixed_precision (bool): whether to use mixed precision
        fp16_allreduce (bool): invoke all reduce on fp16 parameters, only applicable when mixed precision is True
    """
    global comm_handle, is_initialized, computation_dtype, _fp16_all_reduce
    comm_handle = communication_handle(G_inter, G_data, gpus_per_node)
    config.G_inter = G_inter
    config.G_data = G_data
    config.inter_layer_parallel_rank = comm_handle.inter_layer_parallel_rank
    config.data_parallel_rank = comm_handle.data_parallel_rank
    is_initialized = True
    # assert mixed_precision, "Only supports mixed precision at apex O2 level"
    # assert fp16_allreduce, "Only supports fp-16 allreduce"
    if mixed_precision:
        computation_dtype = torch.float16
    else:
        computation_dtype = torch.float32
    _fp16_all_reduce = fp16_allreduce
    if comm_handle.world_rank == 0:
        print(f"Running with G_data={config.G_data} X G_inter={config.G_inter}")


def create_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    micro_batch_size: int,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """
    Create dataloaders for each GPU. For inter_layer_parallel_rank > 0,
    this creates a proxy dataloader which returns zero tensors.

    Arguments:
        dataset (torch.utils.data.Dataset): a PyTorch dataset object
        batch_size (int): batch size for dataloading
        micro_batch_size (int): microbatch size for inter-layer parallelism
        num_workers (int): number of worker processes in the dataloader

    Returns:
        data_loader (torch.utils.data.DataLoader): the dataloader object which
        is a true dataloader for inter_layer_parallel_rank = 0, else it is
        a proxy dataloader
    """
    assert is_initialized
    config.micro_batch_size = micro_batch_size
    config.batch_size = batch_size
    config.batch_size_per_network = batch_size // config.G_data
    assert (
        batch_size % (config.G_data * micro_batch_size) == 0
    ), "Batch Size should be divisible by the G_data*micro_batch_size"

    if config.inter_layer_parallel_rank == 0:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=config.G_data, rank=config.data_parallel_rank
        )
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.batch_size_per_network,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
        )  # not working with drop_last=False

    else:
        dataset = empty_dataset(len(dataset), len(dataset[0]))
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=config.G_data, rank=config.data_parallel_rank
        )
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.batch_size_per_network,
            shuffle=False,
            num_workers=0,
            sampler=sampler,
            drop_last=True,
        )
    return data_loader


def _coalesce_and_reassign(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Coalesce tensors into a flattened 1D tensor and reassign them to
    subtensors in this 1D tensor.

    Arguments:
        tensors (List[torch.Tensor]): list of tensors to be coalesced

    Returns:
        flatenned_tensors (torch.tensor): the flattened tensor.

    """
    flattened_tensor = _flatten_dense_tensors(tensors)
    for old_tensor, new_tensor in zip(
        tensors, _unflatten_dense_tensors(flattened_tensor, tensors)
    ):
        old_tensor.data = new_tensor
    return flattened_tensor


def register_model_and_optimizer(model_shard, optimizer):
    """AxoNN's user facing function to register a model shard and the corresponding optimizer.

    Arguments:
        model_shard (torch.nn.Module): the model shard created by the
        user to be registered
        optimizer  (torch.nn.Optim): optimizer object for the model
    """
    global model, model_params_fp32, model_grads_fp32, model_params_fp16, model_grads_fp16, fp32_optimizer
    assert is_initialized
    # assert computation_dtype == torch.float16, f"Only mixed precision supported - {computation_dtype}"
    if computation_dtype == torch.float16:
        opt_level = 'O2'
    else:
        opt_level = 'O0'

    model = model_shard
    model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=opt_level, loss_scale='dynamic')
    optimizer._amp_lazy_init() # this typecasts all parameters 
                               # in all optimizer groups to fp-32, 
                               # but does not change the model parameters
                               # also recasts the optimzier states by calling self.load_state_dict(self.state_dict())

    stash = optimizer._amp_stash

    if computation_dtype == torch.float16:
        model_params = stash.all_fp16_params # these are fp-16 model parameters
        optimizer_params = stash.all_fp32_from_fp16_params # these are fp-32 optimizer parameters
    else:
        model_params = stash.all_fp32_params
        optimizer_params = stash.all_fp32_params
        assert not stash.all_fp16_params, "all parameters must be fp-32"
        # Here model_params and optimizer_params are same, model_params[0] is optimizer_params[0] should return True

    model_params = _coalesce_and_reassign(model_params)
    comm_handle.allreduce(
        model_params/config.G_data, async_op=False
    )  # sync all parameters across data parallel ranks
    model_grads = []
    for param in model.parameters():
        param.grad = torch.empty_like(param)
        model_grads.append(param.grad)
    model_grads = _coalesce_and_reassign(model_grads)
    fp32_optimizer = optimizer

    if computation_dtype == torch.float16:
        model_params_fp16 = model_params
        model_grads_fp16 = model_grads
    else:
        model_params_fp32 = model_params
        model_grads_fp32 = model_grads
    
    return model, optimizer
    #if computation_dtype == torch.float32:
    #    model_params_fp32 = model_params
    #    model_grads_fp32 = model_grads
    #    model_params_fp16 = model_grads_fp16 = None
    #elif computation_dtype == torch.float16:
    #    print(model_params.dtype, model_grads.dtype)
    #    model_params_fp16 = model_params
    #    model_grads_fp16 = model_grads
    #    model_params_fp32 = model_params.float()
    #    model_grads_fp32 = model_grads.float()

    #print_status(f"Number of params - {torch.numel(model_params)}")
    #print_status(f"{model.dtype}")


def register_loss_fn(loss_fn):
    """AxoNN's user facing function to register a loss function.

    Arguments:
        loss_fn: a PyTorch loss function (eg: torch.nn.CrossEntropy)
    """
    global criterion
    assert is_initialized
    criterion = loss_fn


def _get_subtensor(tensor, microbatch_no):
    """divide the tensor into equal tensors of micro_batch_size and
    retrieve the microbatch_no tensor. Useful when fetching data
    corresponding to a microbatch from a batch/labels.

    Arguments:
        tensor (torch.Tensor): tensor to be divided
    """
    start = microbatch_no * config.micro_batch_size
    end = (microbatch_no + 1) * config.micro_batch_size
    return tensor[start:end]


def print_status(msg):
    """print msg

    Arguments:
        msg (str): message to be printed
    """
    print(
        f"DP Rank : {config.data_parallel_rank} |\
ILP Rank : {config.inter_layer_parallel_rank} - {msg}"
    )


def _forward_pass(input_activation: torch.Tensor, microbatch_no: int):
    """do the forward pass on an input activation and send the data to a forward GPU

    Arguments:
        input_activation (torch.Tensor): input activation from the previous GPU
        microbatch_no (int): the microbatch number of the input activation

    """
    output_activation = model(input_activation)
    input_tensors_cache[microbatch_no] = input_activation
    output_tensors_cache[microbatch_no] = output_activation
    if config.inter_layer_parallel_rank + 1 < config.G_inter:
        _send(output_activation, config.inter_layer_parallel_rank + 1, microbatch_no)


def _clear_transit_tensors(clear_all=False):
    """test pending isends for completion and delete tensors that have been sent

    Arguments:
        clear_all (bool): if true, return only after all isends have finished
    """
    global transit_tensors
    remaining_tensors = []
    for f, tensor in transit_tensors:
        if clear_all:
            f.Wait()
        elif not f.Test():
            remaining_tensors.append([f, tensor])
    transit_tensors = remaining_tensors


def _send(tensor: torch.Tensor, destination: int, tag: int):
    """send a tensor to a particular rank with a particular tag using MPI

    Arguments:
        tensor (torch.Tensor): tensor to be sent
        destination (int): inter-layer-parallel rank of the destination
        tag (int): tag of the message
    """
    if (destination < 0) or (destination >= config.G_inter):
        return
    _clear_transit_tensors()
    tensor = tensor.contiguous()
    torch.cuda.synchronize()
    transit_tensors.append([comm_handle.send(tensor, destination, tag), tensor])


def _post_fw_recv_requests():
    if (requests["fw"] is None) and config.inter_layer_parallel_rank > 0:
        tensor = torch.cuda.FloatTensor(
            size=[config.micro_batch_size] + model.get_input_shape()
        )
        tensor.requires_grad = True
        requests["fw"] = [
            tensor,
            comm_handle.recv(tensor, config.inter_layer_parallel_rank - 1),
        ]


def _post_bw_recv_requests():
    if (requests["bw"] is None) and (
        config.inter_layer_parallel_rank < config.G_inter - 1
    ):
        tensor = torch.cuda.FloatTensor(
            size=[config.micro_batch_size] + model.get_output_shape()
        )
        requests["bw"] = [
            tensor,
            comm_handle.recv(tensor, config.inter_layer_parallel_rank + 1),
        ]


def _post_recv_requests():
    """
    post mpi irecv requests if they haven't been posted.
    """
    _post_fw_recv_requests()
    _post_bw_recv_requests()


def _recv(post_fw_recv=True, post_bw_recv=True) -> int:
    """
    Message driven scheduling of forward and backward passes for pipelining.

    Arguments:
        post_new_requests(bool): Whether to post new receive requests

    Returns:
        tag(int): the tag of the received message which is the microbatch number
    """
    status = MPI.Status()
    if (requests["bw"] is None) and (requests["fw"] is not None):
        requests["fw"][1].Wait(status)
        tag = status.Get_tag()
        input_activation = requests["fw"][0]
        requests["fw"] = None
        if post_fw_recv:
            _post_fw_recv_requests()
        _forward_pass(input_activation, tag)
        op = Operation.FW
    elif (requests["fw"] is None) and (requests["bw"] is not None):
        requests["bw"][1].Wait(status)
        tag = status.Get_tag()
        output_gradients = requests["bw"][0]
        requests["bw"] = None
        if post_bw_recv:
            _post_bw_recv_requests()
        _backward_pass(output_gradients, tag)
        op = Operation.BW
    else:
        index = MPI.Request.Waitany([requests["fw"][1], requests["bw"][1]], status)
        tag = status.Get_tag()
        if index == 0:  # forward pass
            input_activation = requests["fw"][0]
            requests["fw"] = None
            if post_fw_recv:
                _post_fw_recv_requests()
            _forward_pass(input_activation, tag)
            op = Operation.FW
        else:
            output_gradients = requests["bw"][0]
            requests["bw"] = None
            if post_bw_recv:
                _post_bw_recv_requests()
            _backward_pass(output_gradients, tag)
            op = Operation.BW
    return tag, op


def _calc_loss(microbatch_no, microbatch_labels):
    """Calculate the loss for a given microbatch number and its corresponding labels

    Arguments:
        microbatch_no (int): the microbatch number
        microbatch_labels (torch.Tensor): the true labels for the microbatch
    """
    return criterion(output_tensors_cache[microbatch_no], microbatch_labels.view(-1))


def _backward_pass(output_gradients, microbatch_no):
    """do the backward pass of a microbatch and send the input activation gradients
    to the previous GPU.

    Arguments:
        output gradients (torch.Tensor): the gradient of the loss wrt the output tensor
        microbatch_no (int): the microbatch number
    """
    with amp.scale_loss(output_tensors_cache[microbatch_no], fp32_optimizer) as scaled_loss:
        output_tensors_cache[microbatch_no].backward(output_gradients)
    input_tensor = input_tensors_cache[microbatch_no]
    del output_tensors_cache[microbatch_no]
    del input_tensors_cache[microbatch_no]
    if config.inter_layer_parallel_rank - 1 >= 0:
        _send(input_tensor.grad, config.inter_layer_parallel_rank - 1, microbatch_no)


def run_batch(batch: torch.Tensor, labels: torch.Tensor) -> int:
    """Perform forward and backward pass on a batch. This function invokes
    inter-layer-parallelism followed by an all-reduce.

    Arguments:
        batch (torch.Tensor): the input batch, for inter-layer-parallel-rank > 0
        this can be a proxy tensor with the first dimension equal to the batch size
        labels (torch.Tensor): the true labels, for inter-layer-parallel-rank
        < G_inter-1, this can be None

    Returns:
        loss (int): the loss on the batch for inter-layer-parallel-rank
        == G_inter - 1, else 0
    """
    batch_loss = 0
    ilp_rank, G_inter, G_data = (
        config.inter_layer_parallel_rank,
        config.G_inter,
        config.G_data,
    )
    num_microbatches_per_network = batch.shape[0] // config.micro_batch_size
    assert num_microbatches_per_network == 1
    if G_inter == 1:
        for microbatch_no in range(num_microbatches_per_network):
            _forward_pass(_get_subtensor(batch, microbatch_no), microbatch_no)
            microbatch_loss = _calc_loss(
                microbatch_no, _get_subtensor(labels, microbatch_no)
            )
            batch_loss += microbatch_loss.item()
            output_tensors_cache[microbatch_no] = microbatch_loss
            _backward_pass(None, microbatch_no)
    else:
        remaining_microbatches = num_microbatches_per_network
        num_msgs = remaining_microbatches
        if (ilp_rank != 0) and (ilp_rank != G_inter - 1):
            num_msgs += remaining_microbatches
            forward_msgs = backward_msgs = num_msgs // 2
        elif ilp_rank == 0:
            backward_msgs = num_msgs
            forward_msgs = 0
        else:
            forward_msgs = num_msgs
            backward_msgs = 0

        next_microbatch = 0
        if ilp_rank == 0:
            for _ in range(G_inter):
                if remaining_microbatches == 0:
                    break
                _forward_pass(_get_subtensor(batch, next_microbatch), next_microbatch)
                next_microbatch += 1
                remaining_microbatches -= 1

        _post_recv_requests()
        while num_msgs:
            microbatch_no, op = _recv(
                post_fw_recv=(forward_msgs > 1), post_bw_recv=(backward_msgs > 1)
            )
            num_msgs -= 1
            if op == Operation.FW:
                forward_msgs -= 1
            elif op == Operation.BW:
                backward_msgs -= 1
            if ilp_rank == 0 and remaining_microbatches:  # inject next microbatch
                _forward_pass(_get_subtensor(batch, next_microbatch), next_microbatch)
                next_microbatch += 1
                remaining_microbatches -= 1
            elif ilp_rank == G_inter - 1:
                microbatch_loss = _calc_loss(
                    microbatch_no, _get_subtensor(labels, microbatch_no)
                )
                batch_loss += microbatch_loss.item()
                output_tensors_cache[microbatch_no] = microbatch_loss
                _backward_pass(None, microbatch_no)

    if computation_dtype == torch.float16:
        grads = model_grads_fp16
    else:
        grads = model_grads_fp32
    comm_handle.allreduce(
        grads / G_data / num_microbatches_per_network, async_op=False
    )
    return batch_loss / num_microbatches_per_network
