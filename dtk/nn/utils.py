from math import ceil
import torch
import torch.nn.functional as F
import torch.nn as nn
import random



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

    seq = pad_both_ends(seq, pad_left, pad_right, dim=cut_dim)

    stacked = seq.narrow(cut_dim, 0, snip_length).unsqueeze(0)
    iterations = (seq.size()[cut_dim] - snip_length) // cutting_stride + 1
    for i in range(1, iterations):
        stacked = torch.cat((stacked, seq.narrow(cut_dim, i * cutting_stride, snip_length).unsqueeze(0)))
    return stacked


def create_windowed_sequence(seqs, snip_length, cut_dim=0, cutting_stride=None, pad_samples=0):
    windowed_seqs = []
    lengths = []
    for seq in seqs:
        windowed_seqs.append(cut_n_stack(seq, snip_length, cut_dim, cutting_stride, pad_samples).unsqueeze(0))
        lengths.append(windowed_seqs[-1].size(1))

    return torch.cat(windowed_seqs), lengths


def pad_n_stack_sequences(seq_list, order=None, max_length=None):
    # We assume that sequences are provided time x samples
    sizes = [x.size()[0] for x in seq_list]  # Take the length of the sequnece
    if max_length is None:
        max_length = max(sizes)

    tensors = []
    lengths = []

    if order is None:
        indexes = range(0, len(sizes))
        new_order = []
        zipped = zip(sizes, seq_list, indexes)
        for item in sorted(zipped, key=lambda x: x[0], reverse=True):
            size, seq, index = item
            if size > max_length:
                seq = seq[:(max_length - size)]
                size = max_length
            elif size < max_length:
                seq = pad(seq, max_length)

            lengths.append(size)
            tensors.append(seq.unsqueeze(0))
            new_order.append(index)

        return torch.cat(tensors), lengths, new_order
    else:
        for idx in order:
            size = sizes[idx]
            seq = seq_list[idx]
            if size > max_length:
                seq = seq[:(max_length - size)]
                size = max_length
            elif size < max_length:
                seq = pad(seq, max_length)

            lengths.append(size)
            tensors.append(seq.unsqueeze(0))

        return torch.cat(tensors), lengths


def get_seq_output(batch, lengths, feature_size):
    adjusted_lengths = [i * lengths[0] + l for i, l in enumerate(lengths)]
    batch_r = batch.view(-1, feature_size)
    return batch_r.index_select(0, torch.LongTensor(adjusted_lengths).to(batch_r.device) - 1).to(batch_r.device)


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def initialize_weights(net, initialisation=None, bias=None):
    for m in net.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) \
                or isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.ConvTranspose2d) \
                or isinstance(m, nn.ConvTranspose3d):
            if initialisation is None:
                torch.nn.init.xavier_normal_(m.weight)
            else:
                m.weight.data.normal_(initialisation[0], initialisation[1])

            if bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.GRU) or isinstance(m, nn.LSTM):
            for layer_params in m._all_weights:
                for param in layer_params:
                    if 'weight' in param:
                        if initialisation is None:
                            nn.init.xavier_normal_(m._parameters[param])
                        else:
                            nn.init.normal_(m._parameters[param], initialisation[0], initialisation[1])


def same_padding(kernel_size, stride=1, in_size=0):
    out_size = ceil(float(in_size) / float(stride))
    return int((out_size - 1) * stride + kernel_size - in_size)


def calculate_output_size(in_size, kernel_size, stride, padding):
    return int((in_size + padding - kernel_size) / stride) + 1


def calculate_receptive_field(kernels, strides, jump=1, receptive=1):
    for s, k in zip(strides, kernels):
        receptive = receptive + (k - 1) * jump
        jump = jump * s
    return receptive


def subsample_batch(tensor, sample_size, lengths=None):
    batch_size = tensor.size(0)
    if lengths is None:
        lengths = batch_size * [tensor.size(1)]

    tensor_list = []
    for i in range(batch_size):
        start = random.randint(0, lengths[i] - sample_size)
        tensor_list.append(tensor[i, start:start + sample_size])

    return torch.stack(tensor_list)


def broadcast_elements(batch, repeat_no):
    total_tensors = []
    for i in range(0, batch.size()[0]):
        total_tensors += [torch.stack(repeat_no * [batch[i]])]

    return torch.stack(total_tensors)


def model_size(model, only_trainable=False):
    if only_trainable:
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())
