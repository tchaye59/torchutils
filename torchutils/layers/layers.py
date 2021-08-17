from torch import nn


class Lambda(nn.Module):
    def __init__(self, loss_fn):
        super(Lambda, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, pred, target):
        return self.loss_fn(pred, target)


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps
        size = input_seq.size()
        resh_size = size[0] * size[1], *size[2:]
        reshaped_input = input_seq.contiguous().view(resh_size)

        output = self.module(reshaped_input)
        # We have to reshape Y
        # (samples, timesteps, output_size)
        final_size = size[0], size[1], *output.size()[1:]
        output = output.contiguous().view(final_size)
        return output
