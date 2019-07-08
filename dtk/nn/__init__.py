import torch
import torch.nn.functional as F


def pad_both_ends(tensor, left, right, dim=0):
    no_dims = len(tensor.size())
    if dim == -1:
        dim = no_dims - 1

    padding = [0] * 2 * no_dims
    padding[2 * (no_dims - dim - 1)] = left
    padding[2 * (no_dims - dim - 1) + 1] = right
    return F.pad(tensor, padding, "constant", 0)


def pad(tensor, length, dim=0):
    no_dims = len(tensor.size())
    if dim == -1:
        dim = no_dims - 1

    padding = [0] * 2 * no_dims
    padding[2 * (no_dims - dim - 1) + 1] = max(length - tensor.size(dim), 0)
    return F.pad(tensor, padding, "constant", 0)


def cut_n_stack(seq, snip_length, cut_dim=0, cutting_stride=None, pad_samples=0):
    if cutting_stride is None:
        cutting_stride = snip_length

    pad_left = pad_samples // 2
    pad_right = pad_samples - pad_samples // 2

    seq = pad_both_ends(seq, pad_left, pad_right)

    stacked = seq.narrow(cut_dim, 0, snip_length).unsqueeze(0)
    iterations = (seq.size()[cut_dim] - snip_length) // cutting_stride + 1
    for i in range(1, iterations):
        stacked = torch.cat((stacked, seq.narrow(cut_dim, i * cutting_stride, snip_length).unsqueeze(0)))
    return stacked


def get_seq_output(batch, lengths, feature_size):
    adjusted_lengths = [i * lengths[0] + l for i, l in enumerate(lengths)]
    batch_r = batch.view(-1, feature_size)
    return batch_r.index_select(0, torch.LongTensor(adjusted_lengths).to(batch_r.device) - 1).to(batch_r.device)
