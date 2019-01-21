import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class LinearTester(torch.nn.Module):

    def __init__(self, input_size, output_size, layers=3, init_weights=True, gpu_id=None):
        super(LinearTester, self).__init__()
        self.layers = layers
        # size = [C,H,W]
        self.input_size = input_size
        self.output_size = output_size
        self.gpu_id = gpu_id
        self._make_nonLinearLayers()
        self._make_linearLayers()
        if init_weights:
            self._initialize_weights()
        # For record
        self.nonLinearLayersRecord = torch.zeros((layers - 1, *self.output_size)).cuda(gpu_id)

    def forward(self, x):
        out = self.linear(0, x, torch.zeros_like(x))
        for i in range(1, self.layers):
            out = self.nonLinear(i - 1, out)
            out = self.linear(i, x, out)
        return out

    def _make_linearLayers(self):
        self.linearLayers_bn = []
        self.linearLayers_conv = []
        inCh = self.input_size[0]
        outCh = self.output_size[0]
        for x in range(self.layers):
            self.linearLayers_bn += [nn.BatchNorm2d(inCh, affine=True, track_running_stats=True)]
            self.linearLayers_conv += [nn.Conv2d(inCh, outCh, kernel_size=3, padding=1, bias=True)]
        self.linearLayers_bn = nn.ModuleList(self.linearLayers_bn)
        self.linearLayers_conv = nn.ModuleList(self.linearLayers_conv)

    def _make_nonLinearLayers(self):
        self.nonLinearLayers_bn = []
        self.nonLinearLayers_ReLU = []
        outCh = self.output_size[0]
        for x in range(self.layers - 1):
            self.nonLinearLayers_bn += [nn.BatchNorm2d(outCh,affine=True,track_running_stats=True)]
            self.nonLinearLayers_ReLU += [nn.ReLU(inplace=True)]
        self.nonLinearLayers_ReLU = nn.ModuleList(self.nonLinearLayers_ReLU)
        self.nonLinearLayers_bn = nn.ModuleList(self.nonLinearLayers_bn)

    def nonLinear(self, i, out):
        out = self.nonLinearLayers_ReLU[i](out)
        out = self.nonLinearLayers_bn[i](out)
        return out

    def linear(self, i, x, out):
        out = self.linearLayers_bn[i](x) + out
        out = self.linearLayers_conv[i](out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # Useful functions

    def val_linearity(self, x):
        with torch.no_grad():
            Yn_length = torch.zeros(self.layers)
            Yn = torch.zeros((self.layers, *self.output_size))

            if x.size(0) != 1:
                print("Only single mode!")
                return
            # record nonLinearLayersRecord
            z = torch.zeros_like(x).cuda(self.gpu_id)
            # Y_sum
            Y_sum = self.linear(0, x, z).cuda(self.gpu_id)
            for i in range(1, self.layers):
                Y_sum = self._rec_nonLinear(i - 1, Y_sum)
                Y_sum = self.linear(i, x, Y_sum)
            Y_sum = Y_sum.reshape(self.output_size).cpu()
            # Y_zero
            Y_zero = self.linear(0, z, z).cuda(self.gpu_id)
            for i in range(1, self.layers):
                Y_zero = self._yn_nonLinear(i - 1, Y_zero)
                Y_zero = self.linear(i, z, Y_zero)
            Y_zero = Y_zero.reshape(self.output_size).cpu()
            # record Yn
            for n in range(self.layers):
                out = z
                if n == self.layers - 1:
                    out = self.linear(0, x, out)
                else:
                    out = self.linear(0, z, out)
                for i in range(1, self.layers):
                    n_ = self.layers - i - 1
                    if n == n_:
                        out = self._yn_nonLinear(i - 1, out)
                        out = self.linear(i, x, out)
                    else:
                        out = self._yn_nonLinear(i - 1, out)
                        out = self.linear(i, z, out)
                Yn[n] = out.reshape(self.output_size).cpu()
                Yn_length[n] = torch.norm(Yn[n] - Y_zero)
            Yn_contribution = Yn_length ** 2 / torch.sum(Yn_length ** 2)

            return Y_sum, Yn, Yn_contribution, Y_zero

    def _rec_nonLinear(self, i, out):
        out = self.nonLinearLayers_ReLU[i](out)
        self.nonLinearLayersRecord[i] = torch.gt(out, 0).reshape(self.input_size)  ##.float()
        out = self.nonLinearLayers_bn[i](out)
        return out

    def _yn_nonLinear(self,i,out):
        out = self.nonLinearLayersRecord[i] * out
        out = self.nonLinearLayers_bn[i](out)
        return out