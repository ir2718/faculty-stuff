import math
import torch
import torch.nn as nn
import numpy as np

def normalize_img(img, image_mean, image_std):
    return ((img / 255. - image_mean) / image_std).permute(2, 0, 1)

class ConvNormActBlock(torch.nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        norm_layer=None,
        activation_layer=None,
    ):
        super().__init__()
        padding = 'same'
        bias = norm_layer is None
        self.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
            )
        )
        if norm_layer is not None:
            self.append(norm_layer(out_channels))
        if activation_layer is not None:
            self.append(activation_layer())


def decode_boxes(rel_codes, boxes, weights=(1.0, 1.0, 1.0, 1.0), bbox_xform_clip=math.log(1000.0 / 16)):
    """
    Apply predicted bounding-box regression deltas to given roi boxes.
    :param rel_codes: tensor with shape Nx4 containing the encoded offsets t_x, t_y, t_w, t_h
    :param boxes: tensor with shape Nx4 containing the anchor boxes in the format (x1, y1, x2, y2) in absolute
    coordinates (i.e. x in the range [0, width] and y in the range [0, height])
    :param weights: During training, the encoded offsets are multiplied by weights to balance the regression loss.
    Ignore for now :)
    :param bbox_xform_clip: Clipping because of the exponential function. Ignore for now :)
    :return: decoded bounding boxes in the format (x1, y1, x2, y2) in absolute coordinates
    """
    # Unpacking and weighting predicted transformations.
    wx, wy, ww, wh = weights
    tx = rel_codes[:, 0] / wx
    ty = rel_codes[:, 1] / wy
    tw = rel_codes[:, 2] / ww
    th = rel_codes[:, 3] / wh
    # Prevent sending too large values into torch.exp()
    tw = torch.clamp(tw, max=bbox_xform_clip)
    th = torch.clamp(th, max=bbox_xform_clip)

    boxes = boxes.to(rel_codes.dtype)

    # YOUR CODE HERE
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = (boxes[:, 0] + boxes[:, 2]) / 2
    ctr_y = (boxes[:, 1] + boxes[:, 3]) / 2

    preds_x = tx * widths + ctr_x
    preds_y = ty * heights + ctr_y
    preds_w = torch.exp(tw) * widths
    preds_h = torch.exp(th) * heights

    # Apply transformations...
    decoded_boxes = torch.stack([
        preds_x - 0.5 * preds_w, 
        preds_y - 0.5 * preds_h, 
        preds_x + 0.5 * preds_w, 
        preds_y + 0.5 * preds_h
    ], axis=1)


    return decoded_boxes