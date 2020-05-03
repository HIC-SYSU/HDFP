

class Config:
    image_size = 256   #src image size wd = ht
    image_channel = 3  #src image channel
    data_save_path = "./"
    train_path_1 = "/home/amax/DUTS-TR" #train main dir path
    train_images_path_1 = "DUTS-TR-Image"
    train_masks_path_1 = "DUTS-TR-Mask"

    train_path_2 = "/home/amax/MSRA10K"  # train main dir path
    train_images_path_2 = "imgs"
    train_masks_path_2 = "gt"

    train_path_3 = "/home/amax/DUT-OMRON"  # train main dir path
    train_images_path_3 = "imgs"
    train_masks_path_3 = "gt"

    train_path_4 = "/home/amax/THUR-15k"  # train main dir path
    train_images_path_4 = "imgs"
    train_masks_path_4 = "gt"


    test_path = "/home/amax/DUTS-TE"  #train main dir path
    test_images_path = "DUTS-TE-Image"
    test_masks_path = "DUTS-TE-Mask"