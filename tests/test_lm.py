# Copyright 2021 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from axonn import axonn as ax
import torchvision
from models.transformer import causallm_transformer 
from torchvision.transforms import ToTensor
import torch
from tqdm import tqdm
import os
import time

class empty_dataset(torch.utils.data.Dataset):
    """
    Proxy dataset object for GPUs with inter_layer_parallel_rank > 0
    """

    def __init__(self, length, seq_len, dtype, vocab_size):
        """Constructor for the proxy dataset class
        Arguments:
            length (int): number of datapoints in the dataset
            num_tensors (int): number of tensors per datapoint
        Returns:
            A PyTorch dataset object
        """
        self.length = length
        self.dtype = dtype
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sentence = torch.randint(low=0, high=self.vocab_size, size=(self.seq_len,), dtype=torch.long)
        return sentence[:-1], sentence[1:]

def test_vit_mnist():
    bs_per_gpu = 16
    seq_len = 128
    num_gpus = 6
    bs = num_gpus * bs_per_gpu
    mbs = bs_per_gpu
    epochs = 10
    N, D, H = 12, 768, 12

    ax.init(G_data=num_gpus, G_inter=1, mixed_precision=True)

    ilp_rank = ax.config.inter_layer_parallel_rank
    G_inter = ax.config.G_inter

    model = causallm_transformer(
        24,1024,16, 50257, seq_len, 0.1
    ).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    ax.register_model_and_optimizer(model, optimizer)

    ax.register_loss_fn(torch.nn.CrossEntropyLoss())


    train_dataset = empty_dataset(5000, seq_len, dtype=torch.long, vocab_size=50257)
    train_loader = ax.create_dataloader(train_dataset, bs, mbs, 0)

    for epoch_number in range(epochs):
        epoch_loss = 0
        start_time = time.time()
        for x, y in tqdm(train_loader, disable= not (ilp_rank == 0 and ax.config.data_parallel_rank == 0)):
            if ilp_rank == 0:
                x, y = x.cuda(), y.cuda()
            if G_inter > 1:
                if ilp_rank == 0:
                    ax.comm_handle.send(y, G_inter - 1, tag=0, async_op=False)
                elif ilp_rank == G_inter - 1:
                    y = y.long().cuda()
                    ax.comm_handle.recv(y, 0, tag=0, async_op=False)
            batch_loss = ax.run_batch(x, y)
            optimizer.step()
            epoch_loss += batch_loss
        if ilp_rank == G_inter - 1:
            ax.print_status(
                f"Epoch {epoch_number+1} : epoch loss {epoch_loss/len(train_loader)} time = {time.time() - start_time} s"
            )

test_vit_mnist()
