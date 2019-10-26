import torch
import torch.nn as nn
import torch.nn.functional as F



class Linear_Shift(nn.Linear):


    def forward(self, input):
        try:
            if self.bias is not None:
                return F.linear(input, self.weight + self.weight_shift, self.bias + self.bias_shift)
            else:
                return F.linear(input, self.weight + self.weight_shift, self.bias)
        except AttributeError:
            return F.linear(input, self.weight, self.bias)





class Conv2d_Shift(nn.Conv2d):


    def conv2d_forward(self, input, weight , bias):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        try:
            if self.bias is not None:
                return self.conv2d_forward(input, self.weight + self.weight_shift, self.bias + self.bias_shift)
            else:
                return self.conv2d_forward(input, self.weight + self.weight_shift, self.bias)
        except AttributeError:
            return self.conv2d_forward(input, self.weight, self.bias)
