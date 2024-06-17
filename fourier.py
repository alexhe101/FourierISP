import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
def get_spectrum(img):
    img_fft = np.fft.fft2(img)
    img_abs = np.abs(img_fft)
    img_pha = np.angle(img_fft)
    return img_abs,img_pha #abs amp pha:pha
def hisEqualCOlor(img_):
    img = img_.copy()
    ycrcb = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))
    clahe.apply(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img
def get_mag_spe(img):
    img_fft = np.fft.fft2(img)
    img_fft_shift  = np.fft.fftshift(img_fft)
    return np.uint8(20*np.log(1+np.abs(img_fft_shift)))
def getphase(img):
    img_fft = np.fft.fft2(img)
    return np.angle(img_fft)
def color_equal(img):
    t =img.copy()
    out = img.copy()
    out[:,:,0] = cv2.equalizeHist(t[:,:,0])
    out[:,:,1] = cv2.equalizeHist(t[:,:,1])
    out[:,:,2] = cv2.equalizeHist(t[:,:,2])
    return out
def save_frequency_image(img,name):
    img = getphase(img)
    matrix_min = np.min(img)
    matrix_max = np.max(img)
    img = (img.astype(np.float32) - matrix_min) / (matrix_max - matrix_min)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(name+".png",bbox_inches='tight',pad_inches = 0)

if __name__ == '__main__':
    img_id=393
    raw_base= '/data'
    canon_base = "datasets/ZRR/test/canon"
    huawei_raw_path = os.path.join(raw_base,str(img_id)+".png")
    canon_path = os.path.join(canon_base,str(img_id)+".jpg")
    huawei_raw = cv2.imread(huawei_raw_path)
    canon = cv2.imread(canon_path)
    huawei_raw = cv2.cvtColor(huawei_raw,cv2.COLOR_BGR2GRAY)
    save_frequency_image(huawei_raw,"canonamp")
    # canon_spe =get_mag_spe(canon)
    # plt.imshow(canon_spe,cmap='gray')
    # plt.savefig("cannon_spe.jpg")
    # canon_pha = getphase(canon)
    # plt.imshow(canon_pha,cmap='gray')
    # plt.savefig("canon_pha.jpg")
    # pan_spe = get_mag_spe(pan)
    # gt_spe = get_mag_spe(gt)

    # plt.show()
    # gt_abs,gt_pha = get_spectrum(gt)
    # ms_abs,ms_pha = get_spectrum(ms)
    # pan_abs,pan_pha = get_spectrum(pan)
    # ms_abs,pan_abs = np.fft.fftshift(ms_abs),np.fft.fftshift(pan_abs)
    # pan_abs = np.fft.ifftshift(pan_abs)
    # ms_abs = np.fft.ifftshift(ms_abs)
    # constant_pha = np.mean(1)*np.ones(gt_pha.shape)
    # #ms spec+pan pha
    # rep_pan = ms_abs*(np.e ** (1j * constant_pha)) #ms amp+ pan pha
    # rep_pan_img = np.real(np.fft.ifft2(rep_pan))
    # img_pan = np.uint8(np.clip(rep_pan_img,0,255))
    # img_pan = color_equal(img_pan)
    # rep_ms = pan_abs*(np.e ** (1j * constant_pha)) #ms amp+ pan pha
    # rep_ms_img = np.real(np.fft.ifft2(rep_ms))
    # img_ms = np.uint8(np.clip(rep_ms_img,0,255))
    # img_ms = color_equal(img_ms)
    # constant_pha = np.mean(gt_pha)*np.ones(gt_pha.shape)
    # rep_gt_1 = gt_abs*(np.e ** (1j * constant_pha)) #ms amp+ pan pha
    # rep_ms_gt_1 = np.real(np.fft.ifft2(rep_gt_1))
    # img_gt_1 = np.uint8(np.clip(rep_ms_gt_1,0,255))
    # img_gt_1 = color_equal(img_gt_1)
    #
    # constant_abs = np.mean(gt_abs)*np.ones(gt_abs.shape)
    # rep_gt_2 = constant_abs*(np.e ** (1j * pan_pha)) #ms amp+ pan pha
    # rep_ms_gt_2 = np.real(np.fft.ifft2(rep_gt_2))
    # img_gt_2 = np.uint8(np.clip(rep_ms_gt_2,0,255))
    # img_gt_2 = color_equal(img_gt_2)
    # print(gt_abs)
    # fig,ax = plt.subplots(3,2)
    # ax[0][0].imshow(ms)
    # ax[1][0].imshow(pan)
    # ax[2][0].imshow(gt)

    # ax[0][1].imshow(ms_spe)
    # ax[1][1].imshow(pan_spe)
    # ax[2][1].imshow(gt_spe)

    # plt.show()