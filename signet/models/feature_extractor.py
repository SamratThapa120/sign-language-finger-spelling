import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelLinear(nn.Module):
    def __init__(self,inp_dim,out_dim , bias=False):
        super(ChannelLinear,self).__init__()
        self.layer = nn.Linear(inp_dim,out_dim,bias=bias)
    def forward(self,x):
        return self.layer(torch.transpose(x,1,2)).transpose(1,2)

class ECA(nn.Module):
    def __init__(self, kernel_size=5):
        super(ECA, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.global_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = torch.sigmoid(y)

        return x * y.expand_as(x)



class CausalDWConv1D(nn.Module):
    def __init__(self, 
        in_channels=1, 
        out_channels=1, 
        kernel_size=17,
        dilation_rate=1,
        use_bias=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias

        self.pad = nn.ConstantPad1d((dilation_rate*(kernel_size-1),0), 0)
        self.dw_conv = nn.Conv1d(
                            in_channels,
                            in_channels*out_channels,
                            kernel_size,
                            stride=1,
                            dilation=dilation_rate,
                            padding=0,
                            bias=use_bias,
                            groups=in_channels)  # Depthwise conv in PyTorch uses groups=in_channels

    def forward(self, inputs):
        x = self.pad(inputs)
        x = self.dw_conv(x)
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TransformerBlock(nn.Module):
    def __init__(self,dim=256, num_heads=4, expand=4, attn_dropout=0.2, drop_rate=0.2, activation=Swish()):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout,batch_first=True)
        self.dropout1 = nn.Dropout(drop_rate)
        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim*expand, bias=False)
        self.linear2 = nn.Linear(dim*expand, dim, bias=False)
        self.dropout2 = nn.Dropout(drop_rate)
        self.activation = activation

    def forward(self, inputs):
        inputs = torch.transpose(inputs,1,2)
        x = self.norm1(inputs)

        device = x.device
        mask = torch.triu(torch.ones((x.shape[1], x.shape[1])), diagonal=1).bool().to(device)

        x, _ = self.attn(x, x, x,attn_mask=mask)
        x = self.dropout1(x)
        x = x + inputs
        attn_out = x

        x = self.norm2(x)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.dropout2(x)
        x = x + attn_out
        return torch.transpose(x,1,2)


class Conv1DBlock(nn.Module):
    def __init__(self, 
                 channel_size,
                 kernel_size,
                 dilation_rate=1,
                 drop_rate=0.0,
                 expand_ratio=2,
                 activation='relu'
                ):
        super(Conv1DBlock, self).__init__()
        self.channel_size = channel_size
        self.expand_ratio = expand_ratio
        self.drop_rate = drop_rate
        self.activation_func = F.relu if activation == 'relu' else Swish()

        self.expand_conv = ChannelLinear(self.channel_size, self.channel_size*self.expand_ratio, bias=True)
        self.dwconv = CausalDWConv1D(in_channels=self.channel_size*self.expand_ratio, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.bn = nn.BatchNorm1d(num_features=self.channel_size*self.expand_ratio, momentum=0.95)
        self.eca = ECA()
        self.project_conv = ChannelLinear(self.channel_size*self.expand_ratio, self.channel_size, bias=True)
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        skip = x
        x = self.expand_conv(x)
        x = self.activation_func(x)

        x = self.dwconv(x)
        x = self.bn(x)

        x = self.eca(x)

        x = self.project_conv(x)
        if self.drop_rate > 0:
            x = self.dropout(x)

        if x.shape == skip.shape:
            x += skip
        return x


class Cnn1dMhsaFeatureExtractor(nn.Module):
    def __init__(self, CFG):
        super(Cnn1dMhsaFeatureExtractor, self).__init__()
        
        # self.masking = nn.ConstantPad1d((0, CFG.max_len - CFG.CHANNELS), CFG.PAD[0])
        self.stem_conv = nn.Linear(CFG.CHANNELS, CFG.dim, bias=False)
        self.stem_bn = nn.BatchNorm1d(CFG.dim, momentum=0.95)
        self.dropout = nn.Dropout(0.8)

        self.blocks1 = nn.Sequential(
            Conv1DBlock(CFG.dim, 17, drop_rate=0.2),
            Conv1DBlock(CFG.dim, 17, drop_rate=0.2),
            Conv1DBlock(CFG.dim, 17, drop_rate=0.2),
            TransformerBlock(CFG.dim, expand=2)
        )

        self.blocks2 = nn.Sequential(
            Conv1DBlock(CFG.dim, 17, drop_rate=0.2),
            Conv1DBlock(CFG.dim, 17, drop_rate=0.2),
            Conv1DBlock(CFG.dim, 17, drop_rate=0.2),
            TransformerBlock(CFG.dim, expand=2)
        )

        if CFG.dim == 384:  # for the 4x sized model
            self.blocks3 = nn.Sequential(
                Conv1DBlock(CFG.dim, 17, drop_rate=0.2),
                Conv1DBlock(CFG.dim, 17, drop_rate=0.2),
                Conv1DBlock(CFG.dim, 17, drop_rate=0.2),
                TransformerBlock(CFG.dim, expand=2)
            )
            self.blocks4 = nn.Sequential(
                Conv1DBlock(CFG.dim, 17, drop_rate=0.2),
                Conv1DBlock(CFG.dim, 17, drop_rate=0.2),
                Conv1DBlock(CFG.dim, 17, drop_rate=0.2),
                TransformerBlock(CFG.dim, expand=2)
            )
        else:
            self.blocks3 = nn.Sequential()
            self.blocks4 = nn.Sequential()

        self.top_conv = ChannelLinear(CFG.dim, CFG.dim * 2)
        self.classifier = ChannelLinear(CFG.dim * 2, CFG.NUM_CLASSES)

    def forward(self, x):
        x = torch.transpose(self.stem_conv(x),1,2)
        x = self.stem_bn(x)

        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)

        x = self.top_conv(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return torch.transpose(x,1,2)


