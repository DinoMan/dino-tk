import torch
import torch.nn.functional as F


def pad(tensor, length, dim=1):
    no_dims = len(tensor.size())
    if dim == -1:
        dim = no_dims

    padding = [0] * 2 * no_dims
    padding[2 * no_dims - dim] = max(length - tensor.size(dim - 1), 0)
    print(padding)
    return F.pad(tensor, padding, "constant", 0)


def get_seq_output(batch, lengths, feature_size):
    adjusted_lengths = [i * lengths[0] + l for i, l in enumerate(lengths)]
    batch_r = batch.view(-1, feature_size)
    return batch_r.index_select(0, torch.LongTensor(adjusted_lengths).to(batch_r.device) - 1).to(batch_r.device)
