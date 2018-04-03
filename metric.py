import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torchvision import models


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_flops(model, inputs, forward, multiply_adds=False):
    hooks = []
    list_conv = []
    list_linear = []
    list_bn = []
    list_relu = []
    list_pooling = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
        2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def register_flops(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                hooks.append(net.register_forward_hook(conv_hook))
            if isinstance(net, torch.nn.Linear):
                hooks.append(net.register_forward_hook(linear_hook))
            if isinstance(net, torch.nn.BatchNorm2d):
                hooks.append(net.register_forward_hook(bn_hook))
            if isinstance(net, torch.nn.ReLU):
                hooks.append(net.register_forward_hook(relu_hook))
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                hooks.append(net.register_forward_hook(pooling_hook))
            return

        for c in childrens:
            register_flops(c)

    model.eval()
    register_flops(model)

    if not isinstance(inputs, Variable):
        inputs = Variable(torch.rand(inputs).unsqueeze(0), requires_grad=True)
    # outputs = model(inputs, *args)
    outputs = forward(model, inputs)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    print('  + Number of FLOPs: %.2e' % (total_flops))
    [hook.remove() for hook in hooks]


if __name__ == '__main__':
    print_flops(models.resnet50(), inputs=(3, 224, 224))
