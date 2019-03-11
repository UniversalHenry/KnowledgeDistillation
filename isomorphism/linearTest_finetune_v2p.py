import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

class SigmoidP(torch.nn.Module):
    def __init__(self,gpu_id=0,fix_p=False):
        super(SigmoidP, self).__init__()
        self.gpu_id = gpu_id
        self.p_prep = nn.Parameter(torch.zeros(1).cuda(self.gpu_id), requires_grad=(not fix_p))

    def forward(self, x):
        x = x * (nn.Sigmoid()(self.p_prep))
        return x

    def p(self):
        return nn.Sigmoid()(self.p_prep)

class LinearTester(torch.nn.Module):

    def __init__(self, input_size, output_size, all_layers=3, init_weights=True, gpu_id=None, fix_p=False, affine=False,bn=True):
        super(LinearTester, self).__init__()
        self.layers = 1
        self.all_layers = all_layers
        # size = [C,H,W]
        self.affine = affine
        self.input_size = input_size
        self.output_size = output_size
        self.gpu_id = gpu_id
        self.fix_p = fix_p
        self.bn = bn
        self._make_nonLinearLayers()
        self._make_linearLayers()
        self.train_params = list(self.linearLayers_conv[0].parameters())
        self.train_params += list(self.linearLayers_bn[0].parameters())
        if init_weights:
            self._initialize_weights()
        # For record
        self.nonLinearLayersRecord = torch.zeros((all_layers - 1, *self.output_size)).cuda(gpu_id)

    def add_layer(self):
        self.train_params = list(self.linearLayers_conv[self.layers].parameters())
        self.train_params += list(self.linearLayers_bn[self.layers].parameters())
        self.train_params += list(self.nonLinearLayers_p[self.layers - 1].parameters())
        self.train_params += list(self.nonLinearLayers_bn[self.layers - 1].parameters())
        self.layers += 1

    def forward(self, x):
        out = self.linear(0, x, torch.zeros_like(x))
        for i in range(1, self.layers):
            out = self.nonLinear(i - 1, out)
            out = self.linear(i, x, out)
        return out

    def _make_linearLayers(self):
        inCh = self.input_size[0]
        outCh = self.output_size[0]
        if self.bn:
            self.linearLayers_bn = []
        self.linearLayers_conv = []
        for i in range(self.all_layers):
            if self.bn:
                self.linearLayers_bn += [nn.BatchNorm2d(inCh, affine=self.affine, track_running_stats=True).cuda(self.gpu_id)]
            self.linearLayers_conv += [nn.Conv2d(inCh, outCh, kernel_size=3, padding=1, bias=True).cuda(self.gpu_id)]

    def _make_nonLinearLayers(self):
        inCh = self.input_size[0]
        outCh = self.output_size[0]
        self.nonLinearLayers_p = []
        self.nonLinearLayers_bn = []
        self.nonLinearLayers_ReLU = []
        for i in range(self.all_layers-1):
            self.nonLinearLayers_p += [SigmoidP(self.gpu_id,self.fix_p)]
            self.nonLinearLayers_bn += [nn.BatchNorm2d(outCh, affine=self.affine, track_running_stats=True).cuda(self.gpu_id)]
            self.nonLinearLayers_ReLU += [nn.ReLU(inplace=True).cuda(self.gpu_id)]

    def nonLinear(self, i, out):
        out = self.nonLinearLayers_ReLU[i](out)
        out = self.nonLinearLayers_bn[i](out)
        if not self.fix_p:
            out = self.nonLinearLayers_p[i](out)
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
                Yn_length[n] = torch.sum((Yn[n] - Y_zero) ** 2) ** 0.5
            Yn_contribution = Yn_length ** 2 / torch.sum(Yn_length ** 2)

            return Y_sum, Yn, Yn_contribution, Y_zero

    def _rec_nonLinear(self, i, out):
        out = self.nonLinearLayers_ReLU[i](out)
        self.nonLinearLayersRecord[i] = torch.gt(out, 0).reshape(self.input_size)  ##.float()
        out = self.nonLinearLayers_bn[i](out)
        if not self.fix_p:
            out = self.nonLinearLayers_p[i](out)
        return out

    def _yn_nonLinear(self, i, out):
        out = self.nonLinearLayersRecord[i] * out
        out = self.nonLinearLayers_bn[i](out)
        if not self.fix_p:
            out = self.nonLinearLayers_p[i](out)
        return out