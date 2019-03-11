import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np


class LinearTester(torch.nn.Module):

    def __init__(self, input_size, output_size, layers=3, init_weights=True, gpu_id=None, fix_p=False, affine=False,bn=True):
        super(LinearTester, self).__init__()
        self.layers = layers
        # size = [C,H,W]
        self.affine = affine
        self.input_size = input_size
        self.output_size = output_size
        self.gpu_id = gpu_id
        self.bn = bn
        self.nonLinearLayers_p_pre = nn.Parameter(torch.zeros(self.layers - 1), requires_grad=(not fix_p))
        self.nonLinearLayers_p = self.get_p()

        # def _make_linearLayers(self):
        linearLayers_bn = []
        linearLayers_conv = []
        inCh = self.input_size[0]
        outCh = self.output_size[0]
        for x in range(self.layers):
            linearLayers_bn += [nn.BatchNorm2d(inCh, affine=self.affine, track_running_stats=True)]
            linearLayers_conv += [nn.Conv2d(inCh, outCh, kernel_size=3, padding=1, bias=False)]
        if self.bn:
            self.linearLayers_bn = nn.ModuleList(linearLayers_bn)
        self.linearLayers_conv = nn.ModuleList(linearLayers_conv)

        # def _make_nonLinearLayers(self):
        self.nonLinearLayers_ReLU = []
        for x in range(self.layers - 1):
            self.nonLinearLayers_ReLU += [nn.ReLU(inplace=True)]
        self.nonLinearLayers_ReLU = nn.ModuleList(self.nonLinearLayers_ReLU)
        self.nonLinearLayers_norm = nn.Parameter(torch.ones(self.layers - 1, self.output_size[0]), requires_grad= False)
        self.running_times = nn.Parameter(torch.zeros(self.layers-1, dtype=torch.long), requires_grad=False)

        if init_weights:
            self._initialize_weights()
        # For record
        self.nonLinearLayersRecord = torch.zeros((layers - 1, *self.output_size)).cuda(gpu_id)

    def get_p(self):
        return nn.Sigmoid()(self.nonLinearLayers_p_pre)

    def forward(self, x):
        self.nonLinearLayers_p = self.get_p()
        out = self.linear(0, x, torch.zeros_like(x))
        for i in range(1, self.layers):
            out = self.nonLinear(i - 1, out)
            out = self.linear(i, x, out)
        return out


    def my_bn(self, i, out, momentum = 0.1, eps = 1e-5):
        if self.training:
            a = torch.sum(out ** 2, [0, 2, 3]).reshape(1, out.shape[1]) + eps
            a = torch.sqrt(a / out.shape[0] / out.shape[2] / out.shape[3] ).detach()
            if self.running_times[i] == 0:
                self.nonLinearLayers_norm[i] = a
            else:
                self.nonLinearLayers_norm[i] = (1 - momentum) *  self.nonLinearLayers_norm[i] + momentum * a
            self.running_times[i] += 1
            a_ = a.reshape(1,out.shape[1],1,1)
        else:
            a_ = self.nonLinearLayers_norm[i].reshape(1,out.shape[1],1,1)

        a_ = a_.repeat(out.shape[0],1, out.shape[2], out.shape[3])
        out = out / a_
        return out

    def nonLinear(self, i, out):
        out = self.my_bn(i,out)
        out = self.nonLinearLayers_ReLU[i](out)
        out = self.nonLinearLayers_p[i] * out
        return out

    def linear(self, i, x, out):
        if self.bn:
            out = self.linearLayers_bn[i](x) + out
        else:
            out = x + out
        out = self.linearLayers_conv[i](out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) and m.affine:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # Useful functions

    def val_linearity(self, x):
        with torch.no_grad():
            self.nonLinearLayers_p = self.get_p()
            Yn_length = torch.zeros(self.layers)
            Yn = torch.zeros((self.layers, *self.output_size))

            if x.size(0) != 1:
                print("Only single mode!")
                return
            # record nonLinearLayersRecord
            Y_sum = self.linear(0, x, torch.zeros_like(x)).cuda(self.gpu_id)
            for i in range(1, self.layers):
                Y_sum = self._rec_nonLinear(i - 1, Y_sum)
                Y_sum = self.linear(i, x, Y_sum)
            Y_sum = Y_sum.reshape(self.output_size).cpu()
            # record Yn
            for n in range(self.layers):
                z = torch.zeros_like(x).cuda(self.gpu_id)
                out = z
                if n == self.layers - 1:
                    out = self.linear(0, x, out)
                for i in range(1, self.layers):
                    n_ = self.layers - i - 1
                    if n == n_:
                        out = self._yn_nonLinear(i - 1, out)
                        out = self.linear(i, x, out)
                    elif n > n_:
                        out = self._yn_nonLinear(i - 1, out)
                        out = self._yn_linear(i, out)
                Yn[n] = out.reshape(self.output_size).cpu()
                Yn_length[n] = torch.sum(Yn[n] ** 2) ** 0.5
            Yn_contribution = Yn_length ** 2 / torch.sum(Yn_length ** 2)

            return Y_sum, Yn, Yn_contribution, None

    def _rec_nonLinear(self, i, out):
        out = self.my_bn(i, out)
        out = self.nonLinearLayers_ReLU[i](out)
        self.nonLinearLayersRecord[i] = torch.gt(out, 0).reshape(self.input_size)
        out = self.nonLinearLayers_p[i] * out
        return out

    def _yn_nonLinear(self, i, out):
        out = self.my_bn(i, out)
        out = self.nonLinearLayersRecord[i] * out
        out = self.nonLinearLayers_p[i] * out
        return out

    def _yn_linear(self, i, out):
        out = self.linearLayers_conv[i](out)
        return out