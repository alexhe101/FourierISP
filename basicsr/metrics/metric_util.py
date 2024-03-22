import numpy as np

from basicsr.utils import bgr2ycbcr


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        # img = img[:,:,2]
        # img = np.dot(img, [0.1140, 0.5870, 0.2989])
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.
# def to_gray_scale_bgr(img):
# def to_y_channel(img):

#     """Converts a BGR image to grayscale.

#     Args:
#         img (ndarray): BGR image with range [0, 255].

#     Returns:
#         (ndarray): Grayscale image with range [0, 255] (float type) without rounding.
#     """
#     # 将图像转换为浮点数并归一化到 [0, 1]
#     img = img.astype(np.float32) / 255.
    
#     # 如果输入是 BGR 彩色图像（三通道），则使用加权平均来生成灰度图像
#     if img.ndim == 3 and img.shape[2] == 3:
#         gray_img = np.dot(img, [0.1140, 0.5870, 0.2989])
#     else:
#         # 如果输入已经是单通道图像或灰度图像，则不进行转换
#         gray_img = img

#     # 将灰度图像重新缩放到 [0, 255] 范围（不进行四舍五入）
#     gray_img = gray_img * 255.
#     gray_img = gray_img[...,None]
#     return gray_img
