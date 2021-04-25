"""Transformer style net architecture."""
import torch

class TransformerNet(torch.nn.Module):
    """Transformer style net class."""

    def __init__(self):
        """Init transformer style net."""
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True, track_running_stats=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True, track_running_stats=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True, track_running_stats=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True, track_running_stats=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True, track_running_stats=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, inp):
        """Inference od transformer style net."""
        res = self.relu(self.in1(self.conv1(inp)))
        res = self.relu(self.in2(self.conv2(res)))
        res = self.relu(self.in3(self.conv3(res)))
        res = self.res1(res)
        res = self.res2(res)
        res = self.res3(res)
        res = self.res4(res)
        res = self.res5(res)
        res = self.relu(self.in4(self.deconv1(res)))
        res = self.relu(self.in5(self.deconv2(res)))
        res = self.deconv3(res)
        return res


class ConvLayer(torch.nn.Module):
    """Performs common convolution with reflection padding."""

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        """Conv layer init."""
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        """Conv layer inference."""
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock."""

    def __init__(self, channels):
        """Residual block init."""
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True, track_running_stats=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True, track_running_stats=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """Residual block inference."""
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """Upsamples the input and then does a convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        """Upsample conv layer init."""
        super(UpsampleConvLayer, self).__init__()
        self.upsample = 2
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        """Upsample conv layer inference."""
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
