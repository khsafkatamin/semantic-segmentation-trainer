import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, batch_norm):
        super(ConvBlock, self).__init__()
        padding = kernel_size[0] // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(activation())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, udepth, filters1, kernel_size, activation, batch_norm, dropout):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm
        self.kernel_size = kernel_size

        current_in = in_channels
        for d in range(udepth):
            filters = (2 ** d) * filters1
            block = nn.Sequential(
                ConvBlock(current_in, filters, kernel_size, activation, batch_norm),
                ConvBlock(filters, filters, kernel_size, activation, batch_norm)
            )
            self.encoder_layers.append(block)
            current_in = filters

    def forward(self, x):
        features = []
        for i, block in enumerate(self.encoder_layers):
            x = block(x)
            features.append(x)
            if i < len(self.encoder_layers) - 1:
                x = self.pool(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return features


class Decoder(nn.Module):
    def __init__(self, udepth, filters1, kernel_size, activation, batch_norm, dropout):
        super(Decoder, self).__init__()
        self.blocks = nn.ModuleList()
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm
        self.kernel_size = kernel_size

        for d in reversed(range(udepth - 1)):
            filters = (2 ** d) * filters1
            block = nn.ModuleDict({
                'upconv': nn.ConvTranspose2d(filters * 2, filters, kernel_size=2, stride=2),
                'conv1': ConvBlock(filters * 2, filters, kernel_size, activation, batch_norm),
                'conv2': ConvBlock(filters, filters, kernel_size, activation, batch_norm),
            })
            self.blocks.append(block)

    def forward(self, features):
        x = features[-1]
        for i, block in enumerate(self.blocks):
            enc_feat = features[-(i + 2)]
            x = block['upconv'](x)
            x = torch.cat([x, enc_feat], dim=1)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = block['conv1'](x)
            x = block['conv2'](x)
        return x


class UNet(nn.Module):
    def __init__(self, input_channels, num_classes,
                 udepth=5, filters1=16, kernel_size=(3, 3),
                 activation=nn.ReLU, batch_norm=True, dropout=0.1):
        super(UNet, self).__init__()

        self.encoder = Encoder(input_channels, udepth, filters1, kernel_size, activation, batch_norm, dropout)
        self.decoder = Decoder(udepth, filters1, kernel_size, activation, batch_norm, dropout)

        filters_last = (2 ** 0) * filters1  # final decoder output channels
        self.final_conv = nn.Conv2d(filters_last, num_classes, kernel_size=1)
        

    def forward(self, x):
        features = self.encoder(x)
        x = self.decoder(features)
        x = self.final_conv(x)
        return x
