from __future__ import division
from numpy import ogrid,repeat,newaxis
from skimage import io
import  skimage.transform
import numpy as np
import tensorflow as tf



def upsample_skimage(factor,input_img):
    # Pad with 0 values, similar to how Tensorflow does it.
    # Order=1 is bilinear upsampling
    return skimage.transform.rescale(input_img,
                                     factor,
                                     mode='constant',
                                     cval=0,
                                     order=1)


def get_kernel_size(factor):
    """
        Find the kernel size given the desired factor of upsampling.
        """
    return 2 * factor - factor % 2


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """

    filter_size = get_kernel_size(factor)

    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)
    #print ("upsample_kernel=",upsample_kernel)

    for i in xrange(number_of_classes):
        weights[:, :, i, i] = upsample_kernel

    return weights


def upsample_tf(factor, input_img):
    number_of_classes = input_img.shape[2]

    new_height = input_img.shape[0] * factor
    new_width = input_img.shape[1] * factor

    expanded_img = np.expand_dims(input_img, axis=0)

    upsample_filter_np = bilinear_upsample_weights(factor,
                                                   number_of_classes)
    #print("upsample_filter_np=",upsample_filter_np,upsample_filter_np.shape)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            with tf.device("/cpu:0"):
                upsample_filt_pl = tf.placeholder(tf.float32)
                logits_pl = tf.placeholder(tf.float32)



                res = tf.nn.conv2d_transpose(logits_pl, upsample_filt_pl,
                                             output_shape=[1, new_height, new_width, number_of_classes],
                                             strides=[1, factor, factor, 1])

                final_result = sess.run(res,
                                        feed_dict={upsample_filt_pl: upsample_filter_np,
                                                   logits_pl: expanded_img})

    return final_result.squeeze()


def upsample(input,factor,channel=1):
    # upsample_weight
    upsample_filter_np = bilinear_upsample_weights(factor,
                                                   channel)
    # Convert to a Tensor type
    upsample_filter_tensor = tf.constant(upsample_filter_np)
    down_shape = tf.shape(input)
    # Calculate the ouput size of the upsampled tensor here only has a shape
    up_shape = tf.stack([
        down_shape[0],
        down_shape[1] * factor,
        down_shape[2] * factor,
        down_shape[3]
    ])
    # Perform the upsampling
    upsampled_input = tf.nn.conv2d_transpose(input, upsample_filter_tensor,
                                           output_shape=up_shape,
                                           strides=[1, factor, factor, 1])
    upsampled_input = tf.reshape(upsampled_input, [-1, up_shape[1], up_shape[2], channel])

    return upsampled_input



if __name__ == "__main__":
    imsize = 7
    x, y = ogrid[0:imsize, 0:imsize]  # can simple as ogrid[:imsize,:imsize]

    # print(x,x.shape,y,y.shape)
    # print(newaxis)

    # print (x+y,(x+y).shape) #3X3
    #
    # print((x+y)[...,newaxis],(x+y)[...,newaxis].shape ) #3x3X1

    img = repeat((x + y)[..., newaxis], 3, 2) / float(imsize + imsize)

    # print("img=",img,img.shape)

    io.imshow(img, interpolation='none')

    # print(io.available_plugins)

    upsampleed_img_skimage = upsample_skimage(factor=3, input_img=img)

    # io.imshow(upsampleed_img_skimage,interpolation='none')

    # io.show()
    upsampled_img_tf = upsample_tf(factor=3, input_img=img)

    io.imshow(upsampled_img_tf)
    io.show()
