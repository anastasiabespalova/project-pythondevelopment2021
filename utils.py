"""Some useful util functions."""
import torch
import numpy as np
from transformer_net import TransformerNet

def round64(num):
    """Round up a number so it is divided by 64."""
    return (int(num / 64) + (1 if num % 64 else 0)) * 64

def get_net(weights_path=None):
    """Initialize transformer style net."""
    net = TransformerNet()
    if weights_path:
        weights = torch.load(weights_path)
        net.load_state_dict(weights)
    net.eval()
    return net

def preprocess(img):
    """Input image preprocessing for transformer style net."""
    height, width, chan = img.shape
    h_64, w_64 = round64(height), round64(width)
    img_ = np.zeros((h_64, w_64, chan))
    img_[:height, :width, :] = img
    img = img_.copy().swapaxes(1, 2).swapaxes(0, 1)
    img = torch.Tensor(img)[None, :, :, :]
    return img, height, width

def postprocess(res, height, width):
    """Output image processing for transformer style net."""
    res = res[:, :, :height, :width].detach().clamp(0, 255).numpy()[0]
    return res.transpose(1, 2, 0)

def create_circular_mask(radius):
    """Create circular mask for image manipulation."""
    width, height = 2 * radius + 1, 2 * radius + 1
    center = (int(width / 2), int(height / 2))
    radius_ = min(center[0], center[1], width-center[0], height-center[1])

    y_grid, x_grid = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2)

    mask = dist_from_center <= radius_
    return mask
