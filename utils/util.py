import numpy as np
import cv2 as cv

def ExtractROI(image, image_roi):
    x,y,h,w = image_roi
    img_shape = image.shape
    if len(img_shape) == 3:
        roi = image[y:y+h, x:x+w, :]
    else:
        roi = image[y:y+h, x:x+w]

    return roi

def Split_6(img,params):

    dict_part2img={}
    for part in ['l','r','tl','tr','bl','br']:
        roi_coord=params['roi_{}'.format(part)]
        patch=ExtractROI(img, (roi_coord[0],roi_coord[1], roi_coord[3],roi_coord[2],))
        if part =='l' or part=='r':
            patch = np.rot90(patch).copy()
        dict_part2img[part]=patch

    return dict_part2img


def Merge_6(black_img, dict_part2mask,params):

    final_image = black_img.copy()
    for part in ['l','r','tl','tr','bl','br']:
        roi_coord=params['roi_{}'.format(part)]
        # roi_coord[2] = roi_coord[2]-roi_coord[0]
        # roi_coord[3] = roi_coord[3]-roi_coord[1]
        ReplaceROI(final_image, dict_part2mask[part], (roi_coord[0],roi_coord[1], roi_coord[3],roi_coord[2],))

    return final_image


def ReplaceROI(src_image, patch_image, image_roi):
    x,y,h,w = image_roi
    img_shape = src_image.shape
    if len(img_shape) == 3:
        src_image[y:y+h, x:x+w, :] = patch_image
    else:
        src_image[y:y+h, x:x+w] += patch_image

    src_image[y:y+h, x:x+w] = np.where(src_image[y:y+h, x:x+w] > 0, 255, 0)


def MergeImg(image, mask):
    R = np.zeros_like(mask)
    B = np.zeros_like(mask)
    merged_mask = cv.merge([B, mask, R])
    final_img = image*0.9 + merged_mask*0.1
    return final_img
