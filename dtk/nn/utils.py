from math import ceil
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import os
import collections


class Checkpoint():
    def __init__(self, path, model_name, save_every=3, circular=-1, epoch=1):
        self.path = path
        if not os.path.exists(path) and path != "":
            os.makedirs(path)
        self.model_name = model_name.replace(" ", "_").replace(":", "-").replace("-_", "-")
        self.save_every = save_every
        self.circular = circular
        if self.circular > 0:
            self.checkpoints = collections.deque()
        self.epoch = epoch
        self.Init = True

    def __call__(self, state):
        if self.Init:
            self.Init = False
            self.epoch += 1
            return

        if self.epoch % self.save_every:
            self.epoch += 1
            return

        filename = os.path.join(self.path, self.model_name + "_" + str(self.epoch) + ".dat")

        if self.circular > 0:
            if len(self.checkpoints) >= self.circular:
                os.remove(self.checkpoints.popleft())
            self.checkpoints.append(filename)
        torch.save(state, filename)
        self.epoch += 1


def standardize_state_dict(state_dict):
    for k, v in state_dict.copy().items():
        if "module" in k:
            new_k = k[7:]
            state_dict[new_k] = state_dict.pop(k)


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True


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
    for seq in seqs:
        windowed_seqs.append(cut_n_stack(seq, snip_length, cut_dim, cutting_stride, pad_samples).unsqueeze(0))

    return torch.cat(windowed_seqs)


def variable_length_loss(x, y, lengths, loss_func):
    batch_size = x.size(0)
    loss = []
    for i in range(batch_size):
        loss += [loss_func(x[i, : lengths[i]], y[i, : lengths[i]], reduction="none")]
    return torch.mean(torch.cat(loss, axis=0))


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


def initialization(weights, type='xavier', init=None):
    if type == 'normal':
        if init is None:
            torch.nn.init.normal_(weights)
        else:
            torch.nn.init.normal_(weights, mean=init[0], std=init[1])
    elif type == 'xavier':
        if init is None:
            torch.nn.init.xavier_normal_(weights)
        else:
            torch.nn.init.xavier_normal_(weights, gain=init)
    elif type == 'kaiming':
        torch.nn.init.kaiming_normal_(weights)
    elif type == 'orthogonal':
        if init is None:
            torch.nn.init.orthogonal_(weights)
        else:
            torch.nn.init.orthogonal_(weights, gain=init)
    else:
        raise NotImplementedError('Unknown initialization method')


def initialize_weights(net, type='xavier', init=None, init_bias=False, batchnorm_shift=None):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            initialization(m.weight, type=type, init=init)
            if init_bias and hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1 and batchnorm_shift is not None:
            torch.nn.init.normal_(m.weight, 1.0, batchnorm_shift)
            torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.GRU) or isinstance(m, nn.LSTM):
            for layer_params in m._all_weights:
                for param in layer_params:
                    if 'weight' in param:
                        initialization(m._parameters[param])

    net.apply(init_func)


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


def subsample_batch(tensor, sample_size, indices=None, lengths=None):
    batch_size = tensor.size(0)
    if lengths is None:
        lengths = batch_size * [tensor.size(1)]

    if indices is None:
        indices = [random.randint(0, l - sample_size) for l in lengths]

    tensor_list = []
    for i, idx in enumerate(indices):
        tensor_list.append(tensor[i, idx:idx + sample_size])

    return torch.stack(tensor_list).squeeze(), indices


def broadcast_elements(batch, repeat_no, hard_copy=False):
    batch = batch.unsqueeze(1)
    batch_size = list(batch.size())
    batch_size[1] = repeat_no

    if hard_copy:
        return batch.expand(batch_size).contiguous()

    return batch.expand(batch_size)


def model_size(model, only_trainable=False):
    if only_trainable:
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def crop(images, centres, window):
    b, c, h, w = images.size()
    if isinstance(window, int):
        window_w = window
        window_h = window
    else:
        window_w = window[1]
        window_h = window[0]

    centres = centres.squeeze()
    cropped = []
    for i, image in enumerate(images):
        start_w = int(centres[i, 0].detach().cpu().numpy()) - window_w // 2
        end_w = int(centres[i, 0].detach().cpu().numpy()) + window_w // 2
        start_h = int(centres[i, 1].detach().cpu().numpy()) - window_h // 2
        end_h = int(centres[i, 1].detach().cpu().numpy()) + window_h // 2

        if start_h < 0:
            start_h = 0
            end_h = window_h

        if end_h >= h:
            start_h = h - window_h
            end_h = h

        if start_w < 0:
            start_w = 0
            end_w = window_w

        if end_w >= w:
            start_w = w - window_w
            end_w = w

        cropped.append(image[:, start_h:end_h, start_w:end_w].unsqueeze(0))

    return torch.cat(cropped)


def make_pad_mask(lengths):
    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    maximum_len = int(max(lengths))

    seq_range = torch.arange(0, maximum_len, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maximum_len)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def make_non_pad_mask(lengths):
    return ~make_pad_mask(lengths)


def mask_by_length(xs, lengths, fill=0):
    assert xs.size(0) == len(lengths)
    ret = xs.data.new(*xs.size()).fill_(fill)
    for i, l in enumerate(lengths):
        ret[i, :l] = xs[i, :l]
    return ret
