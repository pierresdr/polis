import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import math

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
 
        self.network = nn.Sequential(*layers)

    def  forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
    
    def forward(self, x, channel_last=True):
        x = self.tcn(x.transpose(1, 2) if channel_last else x)
        return self.linear(x.transpose(1, 2))


###########################################################################


class PositionalEncoding(nn.Module):
    def __init__(self, d_model,  ref=0):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.div_term = torch.exp(torch.arange(1, d_model+1, 2).float() * (-math.log(10000.0) / d_model))
        self.ref = ref

    def forward(self, times):
        times = times - self.ref
        pe = torch.zeros((len(times), self.d_model))
        pe[:, 0::2] = torch.sin(times * self.div_term)

        if self.d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(times * self.div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(times * self.div_term)
        return pe


###########################################################################


class RBFLayer(nn.Module):
    def __init__(self, centers, rbf_sigma, infer_centers, infer_stds):
        super(RBFLayer, self).__init__()
        
        self.centers = torch.Tensor(len(centers))
        self.rbf_sigmas = torch.Tensor(len(centers))
        if infer_centers:
            self.centers = nn.Parameter(self.centers, requires_grad=True)
        if infer_stds:
            self.rbf_sigmas = nn.Parameter(self.rbf_sigmas, requires_grad=True)
           
        with torch.no_grad():
            self.centers.copy_(centers)
            nn.init.constant_(self.rbf_sigmas, rbf_sigma)
        
    def forward(self, x):
        return torch.exp( -(x.view(-1,1)-self.centers)**2 / (2*self.rbf_sigmas**2) )       

class RBFNetwork(nn.Module):
    def __init__(self, centers, rbf_sigma, n_output=1, infer_centers=True, infer_stds=True, init_linear_scale=0.2):
        super(RBFNetwork, self).__init__()
        self.rbf_nodes = len(centers)
        self.rbf_layer = RBFLayer(centers, rbf_sigma, infer_centers, infer_stds)
        self.linear_layer = nn.Linear(self.rbf_nodes, n_output)
        
        with torch.no_grad():
            nn.init.normal_(self.linear_layer.weight, std=init_linear_scale)
            nn.init.normal_(self.linear_layer.bias, std=init_linear_scale)
    
    def forward(self, x):
        x = torch.tensor(x)
        x = self.rbf_layer(x)
        x = self.linear_layer(x)
        return x.squeeze()
