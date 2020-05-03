# -*- coding=utf-8 -*-
import os
import cv2
import numpy as np
from config import Config as cg

"""
image preprogressing resize ro same wd and ht
"""


def image_resize(image=0, image_channel=3):
    ht, wd, ch = image.shape
    new_ht = 0
    new_wd = 0
    imagec = 0
    if ht > wd:
        new_ht = cg.image_size
        new_wd = wd * new_ht / ht
        new_wd = int(new_wd)
        new_ht = int(new_ht)
        pad_zeros = np.zeros((cg.image_size, cg.image_size - new_wd, image_channel), dtype=image.dtype)
        # print(new_ht, " ", new_wd)
        imagec = cv2.resize(image, (new_wd, new_ht))  # make sure max(wd,ht) = cg.image_size
        imagec = np.hstack((imagec, pad_zeros))
    else:
        new_wd = cg.image_size
        new_ht = ht * new_wd / wd
        new_wd = int(new_wd)
        new_ht = int(new_ht)
        pad_zeros = np.zeros((cg.image_size - new_ht, cg.image_size, image_channel), dtype=image.dtype)
        # print(new_ht, " ", new_wd)
        imagec = cv2.resize(image, (new_wd, new_ht))  # make sure max(wd,ht) = cg.image_size
        imagec = np.vstack((imagec, pad_zeros))
    return imagec


"""
load data from one dir
"""


def load_data_one(path, images_path, masks_path,is_augmented=False):
    files_img = os.listdir(path + "/" + images_path)
    files_mask = os.listdir(path + "/" + masks_path)
    print("img number = %i" % len(files_img))
    print("mask number = %i " % len(files_mask))
    images = []
    masks = []
    i = 0
    for file in files_mask:  # 遍历文件夹
        file_path_mask = path + "/" + masks_path + "/" + file
        file = file.replace(".png", ".jpg")
        file_path_img = path + "/" + images_path + "/" + file
        # print(file_path_img, " ", file_path_mask)
        image = cv2.imread(file_path_img)  
        mask = cv2.imread(file_path_mask)
        # print("image dtype=", image.dtype, "\n", "mask dtype=", mask.dtype)
        #print("image shape=", image.shape, "\n", "mask shape=", mask.shape)
        # cv2.imshow(file+"img",image)
        # cv2.imshow(file+"mask", mask)

        # #comput mean and std  these code convert to net file
        # image = np.array(image,dtype=np.float32)
        # image[:, :, 2] -= np.mean(image[:,:,2])
        # image[:, :, 1] -= np.mean(image[:,:,1])
        # image[:, :, 0] -= np.mean(image[:,:,0])
        #
        # image[:, :, 2] /= ( np.std(image[:,:,2]) + 1e-12)
        # image[:, :, 1] /= ( np.std(image[:,:,1]) + 1e-12)
        # image[:, :, 0] /= ( np.std(image[:,:,0]) + 1e-12)

        imagec = image_resize(image, cg.image_channel)
        maskc = image_resize(mask, cg.image_channel)
        maskc = maskc[:, :, 0]  # opencv read in *.png default three channel
        th, maskc = cv2.threshold(maskc, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        images.append(imagec)
        masks.append(maskc)

        if is_augmented:
            imagec_hf = cv2.flip(imagec.copy(),0)
            maskc_hf = cv2.flip(maskc.copy(),0)


            imagec_vf = cv2.flip(imagec.copy(), 1)
            maskc_vf = cv2.flip(maskc.copy(), 1)


            imagec_bf = cv2.flip(imagec.copy(), -1)
            maskc_bf = cv2.flip(maskc.copy(), -1)

            # cv2.imshow(file + "c" + "img", imagec)
            # cv2.imshow(file + "c" + "mask", maskc)
            # cv2.imshow(file + "c_vf" + "img", imagec_vf)
            # cv2.imshow(file + "c_vf" + "mask", maskc_vf)
            # cv2.imshow(file + "c_hf" + "img", imagec_hf)
            # cv2.imshow(file + "c_hf" + "mask", maskc_hf)
            # cv2.imshow(file + "c_bf" + "img", imagec_bf)
            # cv2.imshow(file + "c_bf" + "mask", maskc_bf)

            images.append(imagec_hf)
            masks.append(maskc_hf)

            images.append(imagec_vf)
            masks.append(maskc_vf)

            images.append(imagec_bf)
            masks.append(maskc_bf)

        #cv2.waitKey(0)
        i = i + 1
        print(i)
        # if i > 1:
        #     break
    images = np.array(images,dtype=np.uint8)
    masks = np.array(masks,dtype=np.uint8)

    print("i=",i)

    print("images shape=", images.shape, "\n", "masks shape=", masks.shape)
    print("images dtype=", images.dtype, "\n", "masks dtype=", masks.dtype)



    if is_augmented:
        images = np.reshape(images,[4 * i,cg.image_size * cg.image_size *cg.image_channel])
        masks  = np.reshape(masks,[4 * i,cg.image_size*cg.image_size])
    else:
        images = np.reshape(images, [1 * i, cg.image_size * cg.image_size * cg.image_channel])
        masks = np.reshape(masks, [1 * i, cg.image_size * cg.image_size])

    print("images shape=", images.shape, "\n", "masks shape=", masks.shape)
    print("images dtype=", images.dtype, "\n", "masks dtype=", masks.dtype)
    return images, masks


"""
load train and test data
"""


def load_data():
    print("----------------------")
    print("load train data from %s" % cg.train_path_1)
    train_images_1, train_masks_1 = load_data_one(path=cg.train_path_1, images_path=cg.train_images_path_1,
                                              masks_path=cg.train_masks_path_1,is_augmented=True)

    print("----------------------")
    print("load train data from %s" % cg.train_path_2)
    train_images_2, train_masks_2 = load_data_one(path=cg.train_path_2, images_path=cg.train_images_path_2,
                                              masks_path=cg.train_masks_path_2,is_augmented=True)

    print("----------------------")
    print("load train data from %s" % cg.train_path_3)
    train_images_3, train_masks_3 = load_data_one(path=cg.train_path_3, images_path=cg.train_images_path_3,
                                                  masks_path=cg.train_masks_path_3,is_augmented=True)


    train_images = np.concatenate([train_images_1,train_images_2,train_images_3],0)
    train_masks = np.concatenate([train_masks_1,train_masks_2,train_masks_3],0)

    print("train_images.shape",np.shape(train_images))
    print("train_masks.shape", np.shape(train_masks))

    np.save("train_images", train_images)
    np.save("train_masks", train_masks)

    print("----------------------")
    print("load test data from %s" % cg.test_path)
    test_images, test_masks = load_data_one(path=cg.test_path, images_path=cg.test_images_path,
                                            masks_path=cg.test_masks_path,is_augmented=False)
    np.save("test_images", test_images)
    np.save("test_masks", test_masks)
    return train_images, train_masks, test_images, test_masks


if __name__ == '__main__':
    # load images and masks
    load_data()
