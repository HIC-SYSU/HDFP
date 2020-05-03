# -*- coding=utf-8 -*-

# edit from v51
#without pdc using bn(changle)

import numpy as np
import tensorflow as tf
from config import Config as cg
from upsample_skimage import bilinear_upsample_weights,upsample_tf,upsample
import cv2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
log_dir = "/home/amax/PycharmProjects/log3"

batch_size = 4

#images (n,h*w*c) masks (n,h*w)
def images_preprocessing(images,masks):

    images = np.array(images,dtype=np.float32)
    masks = np.array(masks,dtype=np.float32)

    # print("-------------------before process-----------------")
    # print("images.shape", images.shape, '  ', "masks.shape", masks.shape)
    # print("images.dtype",images.dtype,'  ',"mask.dtype",masks.dtype)
    # print("images.minmax",np.min(images),' ',np.max(images))
    # print("masks.minmax",np.min(masks),' ',np.max(masks))


    #procesing images

    batchs,pixes = np.shape(images)

    images = np.reshape(images,newshape=[batchs,cg.image_size,cg.image_size,cg.image_channel])

    for i in range(batchs):

        #cv2.imshow('src',images[i,:, :,:]/ 255.0)

        images[i,:, :, 2] -= np.mean(images[i,:,:,2])
        images[i,:, :, 1] -= np.mean(images[i,:,:,1])
        images[i,:, :, 0] -= np.mean(images[i,:,:,0])

        images[i,:, :, 2] /= ( np.std(images[i,:,:,2]) + 1e-12)
        images[i,:, :, 1] /= ( np.std(images[i,:,:,1]) + 1e-12)
        images[i,:, :, 0] /= ( np.std(images[i,:,:,0]) + 1e-12)

        # cv2.imshow('processed', images[i, :, :, :])
        #
        # cv2.waitKey()


    images = np.reshape(images,newshape=[batchs,cg.image_size*cg.image_size*cg.image_channel])
    #procesing masks

    masks = masks / 255.0

    # print("-------------------after process-----------------")
    # print("images.shape", images.shape, '  ', "masks.shape", masks.shape)
    # print("images.dtype", images.dtype, '  ', "mask.dtype", masks.dtype)
    # print("images.minmax", np.min(images), ' ', np.max(images))
    # print("masks.minmax", np.min(masks), ' ', np.max(masks))

    return images,masks


def run_in_batch_avg(session, tensors, batch_placeholders, test_summary_writer, epoch, feed_dict={}, batch_size=batch_size):
    res = [0] * (len(tensors) - 1)  # produce len(tensors) zero list *reprent repeat
    batch_tensors = [(placeholder, feed_dict[placeholder]) for placeholder in batch_placeholders]
    # print(batch_tensors) the tuple of (placehoder,data array)
    # print(len(batch_tensors)) 2
    total_size = len(batch_tensors[0][1])  # first placeholder's data array length
    batch_count = (total_size + batch_size - 1) / batch_size
    for batch_idx in range(batch_count):
        current_batch_size = None

        for (placeholder, tensor) in batch_tensors:  # two placehoder's value must be change simultaneously
            batch_tensor = tensor[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            current_batch_size = len(batch_tensor)
            feed_dict[placeholder] = tensor[
                                     batch_idx * batch_size: (batch_idx + 1) * batch_size]  # change the xs ys's data
        #preprocessing images and masks
        feed_dict[batch_placeholders[0]],feed_dict[batch_placeholders[1]] = images_preprocessing(feed_dict[batch_placeholders[0]],feed_dict[batch_placeholders[1]])

        tmp = session.run(tensors, feed_dict=feed_dict)
        test_summary_writer.add_summary(tmp[0], epoch * batch_count + batch_idx)
        res = [r + t * current_batch_size for (r, t) in zip(res, tmp[1:])]  # weghted average
    return [r / float(total_size) for r in res]


def weight_variable(shape):
    # initial = tf.truncated_normal(shape, stddev=0.01)
    # return tf.Variable(initial)
    initial = tf.contrib.layers.xavier_initializer_conv2d()
    return tf.Variable(initial(shape=shape))


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(input, in_features, out_features, kernel_size, with_bias=False):
    W = weight_variable([kernel_size, kernel_size, in_features, out_features])
    conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME')
    if with_bias:
        return conv + bias_variable([out_features])
    return conv

def diated_conv2d(input, in_features, out_features, kernel_size, dilated_rate,with_bias=False):
    W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
    conv = tf.nn.atrous_conv2d(input,W,dilated_rate,padding='SAME')
    if with_bias:
        return conv + bias_variable([ out_features ])
    return conv


def batch_activ_conv_not_dialetd(current, in_features, out_features, kernel_size, is_training, keep_prob):
    current = batchnorm(current,is_training=is_training)
    current = tf.nn.relu(current)
    current = conv2d(current, in_features, out_features, kernel_size)
    current = tf.nn.dropout(current, keep_prob)
    return current


def batch_activ_conv(current, in_features, out_features, kernel_size,dilated_rate, is_training, keep_prob):
    current = batchnorm(current,is_training=is_training)
    current = tf.nn.relu(current)
    current = diated_conv2d(current, in_features, out_features, kernel_size,dilated_rate)
    current = tf.nn.dropout(current, keep_prob)
    return current

def block(input, layers, in_features, growth,dilated_rate, is_training, keep_prob):
    current = input
    features = in_features
    for idx in range(layers):
        tmp = batch_activ_conv(current, features, growth, 3, dilated_rate,is_training, keep_prob)
        current = tf.concat((current, tmp), 3)  # 按照通道堆叠起来
        features += growth
    return current, features


def avg_pool(input, s):
    return tf.nn.avg_pool(input, [1, s, s, 1], [1, s, s, 1], 'VALID')

#pdc
def pyramid_dilated_conv(tensor,features_channel,pyramid_feature_chnanels = 1):
    #1
    pyramid_layer1 = conv2d(tensor, features_channel, pyramid_feature_chnanels, 1)

    #3
    pyramid_layer2 = diated_conv2d(tensor, features_channel, pyramid_feature_chnanels, 3, 2)

    #5
    pyramid_layer3 = diated_conv2d(tensor, features_channel, pyramid_feature_chnanels, 3, 2)
    pyramid_layer3 = diated_conv2d(pyramid_layer3, pyramid_feature_chnanels, pyramid_feature_chnanels, 3, 2)

    #7
    pyramid_layer4 = diated_conv2d(tensor, features_channel, pyramid_feature_chnanels, 3, 2)
    pyramid_layer4 = diated_conv2d(pyramid_layer4, pyramid_feature_chnanels, pyramid_feature_chnanels, 3, 2)
    pyramid_layer4 = diated_conv2d(pyramid_layer4, pyramid_feature_chnanels, pyramid_feature_chnanels, 3, 2)


    #9
    pyramid_layer5 = diated_conv2d(tensor, features_channel, pyramid_feature_chnanels, 3, 2)
    pyramid_layer5 = diated_conv2d(pyramid_layer5, pyramid_feature_chnanels, pyramid_feature_chnanels, 3, 2)
    pyramid_layer5 = diated_conv2d(pyramid_layer5, pyramid_feature_chnanels, pyramid_feature_chnanels, 3, 2)
    pyramid_layer5 = diated_conv2d(pyramid_layer5, pyramid_feature_chnanels, pyramid_feature_chnanels, 3, 2)

    print("pyramid_layer1", pyramid_layer1)
    print("pyramid_layer2", pyramid_layer2)
    print("pyramid_layer3", pyramid_layer3)
    print("pyramid_layer4", pyramid_layer4)
    print("pyramid_layer5", pyramid_layer5)

    pdc = tf.concat((pyramid_layer1, pyramid_layer2, pyramid_layer3, pyramid_layer4, pyramid_layer5), 3)  # 按照通道堆叠起来
    return pdc,pyramid_feature_chnanels*5


#ppm for 64X64
def pyramid_pooling_64(tensor,features_channel,pyramid_feature_chnanels = 1):
    pyramid_layer1 = avg_pool(tensor, 64)
    pyramid_layer1_compressed = conv2d(pyramid_layer1, features_channel, pyramid_feature_chnanels, 1)
    pyramid_layer1_upsampled = upsample(pyramid_layer1_compressed, 64, pyramid_feature_chnanels)

    pyramid_layer2 = avg_pool(tensor,32)
    pyramid_layer2_compressed = conv2d(pyramid_layer2, features_channel, pyramid_feature_chnanels, 1)
    pyramid_layer2_upsampled = upsample(pyramid_layer2_compressed,32,pyramid_feature_chnanels)

    pyramid_layer3 = avg_pool(tensor, 21)
    print("pyramid_layer3",pyramid_layer3)
    pyramid_layer3_compressed = conv2d(pyramid_layer3, features_channel, pyramid_feature_chnanels, 1)
    pyramid_layer3_upsampled = upsample(pyramid_layer3_compressed, 21, pyramid_feature_chnanels)
    pyramid_layer3_upsampled = tf.pad(pyramid_layer3_upsampled,[[0,0],[0,1],[0,1],[0,0]],mode="SYMMETRIC")
    print("pyramid_layer3_upsampled",pyramid_layer3_upsampled)

    pyramid_layer4 = avg_pool(tensor, 10)
    print("pyramid_layer4", pyramid_layer4)
    pyramid_layer4_compressed = conv2d(pyramid_layer4, features_channel, pyramid_feature_chnanels, 1)
    pyramid_layer4_upsampled = upsample(pyramid_layer4_compressed, 10, pyramid_feature_chnanels)
    pyramid_layer4_upsampled = tf.pad(pyramid_layer4_upsampled, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="SYMMETRIC")
    print("pyramid_layer4_upsampled", pyramid_layer4_upsampled)


    pyramid = tf.concat((pyramid_layer1_upsampled,
                         pyramid_layer2_upsampled,
                         pyramid_layer3_upsampled,
                         pyramid_layer4_upsampled),
                        3)  # 按照通道堆叠起来

    pyramid = tf.reshape(pyramid,[-1,64,64,4*pyramid_feature_chnanels])

    return pyramid,pyramid_feature_chnanels*4



def fused_loss(yp,gt):

    mae_loss = tf.reduce_mean(tf.log(1 + tf.exp(tf.abs(yp - gt))))
    tf.summary.scalar("mae_loss",mae_loss)

    mask_front = gt
    mask_background = 1 - gt
    pro_front = yp
    pro_background = 1- yp

    w1 = 1 / ( tf.pow(tf.reduce_sum(mask_front),2) + 1e-12)
    w2 = 1 / ( tf.pow(tf.reduce_sum(mask_background),2) + 1e-12)
    numerator = w1 * tf.reduce_sum(mask_front * pro_front) + w2 * tf.reduce_sum(mask_background * pro_background)
    denominator = w1 * tf.reduce_sum(mask_front + pro_front) + w2 * tf.reduce_sum(mask_background + pro_background)
    dice_loss = 1 - 2*numerator/ (denominator + 1e-12)
    tf.summary.scalar("dice_loss", dice_loss)

    w = (cg.image_size * cg.image_size * batch_size - tf.reduce_sum(yp)) / (tf.reduce_sum(yp) + 1e-12)
    #w = (cg.image_size * cg.image_size * 1 - tf.reduce_sum(yp)) / (tf.reduce_sum(yp) + 1e-12)

    print('w=',w)
    cross_entropy_loss = -tf.reduce_mean( 0.1*w * mask_front * tf.log(pro_front + 1e-12) + mask_background * tf.log(pro_background + 1e-12))
    #cross_entropy_loss = -tf.reduce_mean( mask_front * tf.log(pro_front + 1e-12) + mask_background * tf.log(pro_background + 1e-12))

    tf.summary.scalar("cross_entropy_loss", cross_entropy_loss)

    return  dice_loss + mae_loss + cross_entropy_loss
    #return  mae_loss + cross_entropy_loss


def batchnorm(x,is_training,center=True,scale =True,epsilon = 0.001,decay=0.95):

    shape = x.get_shape().as_list()
    mean, var = tf.nn.moments(x,
                    axes=range(len(shape)-1)# [0] is batch dimension
                )

    ema = tf.train.ExponentialMovingAverage(decay=decay)  # decay of exponential moving average

    def mean_var_with_update():
        ema_apply_op = ema.apply([mean, var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(mean), tf.identity(var)

    #mean, var = mean_var_with_update()
    mean, var = tf.cond(is_training,  # is_training value True/False
                        mean_var_with_update,
                        lambda: (
                            ema.average(mean),
                            ema.average(var)
                        )
                        )
    if center == True:
        shift_v = tf.Variable(tf.zeros(shape[-1]))
    else:
        shift_v = None

    if scale == True:
        scale_v = tf.Variable(tf.ones(shape[-1]))
    else:
        scale_v = None


    output = tf.nn.batch_normalization(x, mean, var, shift_v, scale_v, epsilon)
    return output


# def batchnorm(x,is_training,center=True,scale =True,epsilon = 0.001,decay=0.95):
#
#     output =  tf.contrib.layers.batch_norm(x,
#                                            center=center,
#                                            scale=scale,
#                                            is_training=is_training,
#                                            updates_collections=None,
#                                            decay = decay,
#                                            epsilon = epsilon)
#     return output


def F_measure(gt,map):
    mask = tf.greater(map,0.5)
    mask = tf.cast(mask,tf.float32)

    gtCnt = tf.reduce_sum(gt)

    hitMap = tf.where(gt>0,mask,tf.zeros(tf.shape(mask)))

    hitCnt = tf.reduce_sum(hitMap)
    algCnt = tf.reduce_sum(mask)

    prec = hitCnt / (algCnt + 1e-12)
    recall = hitCnt / (gtCnt + 1e-12)

    beta_square = 0.3
    F_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall + 1e-12)

    return  prec,recall,F_score



def run_model(data, image_dim, label_count):
    weight_decay = 1e-4
    layers = 12  # the number of block is fixed,the number of layer in a block is changeable
    graph = tf.Graph()
    with graph.as_default():
        xs = tf.placeholder("float", shape=[None, image_dim],name="xs")
        ys = tf.placeholder("float", shape=[None, label_count],name="ys")
        ysc = tf.reshape(ys, [-1, cg.image_size, cg.image_size, 1])
        tf.summary.image("ysc", ysc, 100)
        print(ysc)

        lr = tf.placeholder("float", shape=[],name='lr')
        keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        is_training = tf.placeholder("bool", shape=[],name='is_training')

        input = tf.reshape(xs, [-1, cg.image_size, cg.image_size, 3],name='src')
        tf.summary.image("xs", input, 100)

        # pyramid feature
        pyramid_feature_chnanels = 8
        fuse_channals = 16

        theta = 0.5

        current = conv2d(input, 3, 16, 3)

        print("layers",layers)

        #block1 256X256 1
        current, features = block(current, layers, 16, 12, 1,is_training, keep_prob)
        scale_256 = conv2d(current,features, fuse_channals, 3, True)

        current = batch_activ_conv(current, features, np.int32(features*theta), 1,1, is_training, keep_prob)
        current = avg_pool(current, 2)

        #block2 128X128 1
        current, features = block(current, layers, np.int32(features*theta), 12,1, is_training, keep_prob)
        scale_128 = conv2d(current, features, fuse_channals, 3, True)
        current = batch_activ_conv(current, features, np.int32(features*theta), 1,1, is_training, keep_prob)

        current = avg_pool(current, 2)

        #block3 64X64 d = 2
        current, features = block(current, layers, np.int32(features * theta), 12,2, is_training, keep_prob)
        scale_64_1 = conv2d(current,features, fuse_channals, 3, True)
        current = batch_activ_conv(current, features, np.int32(features * theta), 1,1, is_training, keep_prob)


        # block4 64X64 d = 4
        current, features = block(current, layers, np.int32(features * theta), 12, 4,is_training, keep_prob)
        scale_64_2 = conv2d(current, features, fuse_channals, 3, True)
        current = batch_activ_conv(current, features, np.int32(features * theta), 1,1, is_training, keep_prob)



        #block5 64X64 d = 8
        current, features = block(current, layers, np.int32(features*theta), 12,8, is_training, keep_prob)
        scale_64_3 = conv2d(current, features, fuse_channals, 3, True)

        print("feature=", features)


        #64_3 Map

        ppm_64_3, ppm_channals_64_3 = pyramid_pooling_64(scale_64_3, features_channel=fuse_channals,
                                                     pyramid_feature_chnanels=1)

        concat_64_3 = tf.concat([scale_64_3, ppm_64_3], 3, name="concat_pyramid_64_3")

        current_64_3 = batchnorm(concat_64_3,is_training=is_training)

        current_64_3 = tf.nn.relu(current_64_3)

        print("current_64_3", current_64_3)
        concat_64_3_channals = fuse_channals + ppm_channals_64_3
        print("concat_64_3_channals",concat_64_3_channals)
        logits_scale_64_3 = conv2d(current_64_3,concat_64_3_channals , 1, 3)

        # 64_2 Map

        concat_64_2 = tf.concat([scale_64_2,concat_64_3], 3, name="concat_pyramid_64_2")
        current_64_2 = batchnorm(concat_64_2, is_training=is_training)
        current_64_2 = tf.nn.relu(current_64_2)

        print("current_64_2", current_64_2)
        concat_64_2_channals = fuse_channals  + concat_64_3_channals
        print("concat_64_2_channals", concat_64_2_channals)
        logits_scale_64_2 = conv2d(current_64_2, concat_64_2_channals, 1, 3)

        # 64_1 Map


        concat_64_1 = tf.concat([scale_64_1, concat_64_2], 3, name="concat_pyramid_64_1")
        current_64_1 = batchnorm(concat_64_1, is_training=is_training)
        current_64_1 = tf.nn.relu(current_64_1)

        print("current_64_1", current_64_1)
        concat_64_1_channals = fuse_channals  + concat_64_2_channals
        print("concat_64_1_channals", concat_64_1_channals)
        logits_scale_64_1 = conv2d(current_64_1,concat_64_1_channals , 1, 3)


        # recovery 128
        concat_scale_64_upsamped = upsample(concat_64_1, 2, concat_64_1_channals)

        logits_scale_128_concat = tf.concat((scale_128,concat_scale_64_upsamped), 3)

        logits_scale_current_128 = batchnorm(logits_scale_128_concat, is_training=is_training)

        logits_scale_current_128 = tf.nn.relu(logits_scale_current_128)

        concat_128_channals = fuse_channals + concat_64_1_channals
        print("concat_128_channals",concat_128_channals)
        logits_scale_128= conv2d(logits_scale_current_128, concat_128_channals, 1, 3, True)


        # recovery 256
        logits_scale_128_upsamped = upsample(logits_scale_128_concat, 2, concat_128_channals)


        logits_scale_256_concat = tf.concat((scale_256,logits_scale_128_upsamped), 3)  # 按照通道堆叠起来

        logits_scale_current_256 = batchnorm(logits_scale_256_concat,is_training=is_training)

        logits_scale_current_256 = tf.nn.relu(logits_scale_current_256)
        print("logits_scale_current_256",logits_scale_current_256)
        concat_256_channals = fuse_channals + concat_128_channals
        logits_scale_256 = conv2d(logits_scale_current_256,concat_256_channals, 1, 3, True)

        logits_scale_64_3_upsampled_to_256 = upsample(logits_scale_64_3,4,1)
        logits_scale_64_2_upsampled_to_256 = upsample(logits_scale_64_2, 4, 1)
        logits_scale_64_1_upsampled_to_256 = upsample(logits_scale_64_1, 4, 1)
        logits_scale_128_upsampled_to_256 = upsample(logits_scale_128, 2, 1)

        logits_scale_64_3_upsampled_to_256_sigmoid = tf.nn.sigmoid(logits_scale_64_3_upsampled_to_256,'scale_64_3_map')
        tf.summary.image('scale_64_3_map',logits_scale_64_3_upsampled_to_256_sigmoid)
        logits_scale_64_2_upsampled_to_256_sigmoid = tf.nn.sigmoid(logits_scale_64_2_upsampled_to_256,'scale_64_2_map')
        tf.summary.image('scale_64_2_map', logits_scale_64_2_upsampled_to_256_sigmoid)
        logits_scale_64_1_upsampled_to_256_sigmoid = tf.nn.sigmoid(logits_scale_64_1_upsampled_to_256,'scale_64_1_map')
        tf.summary.image('scale_64_1_map', logits_scale_64_1_upsampled_to_256_sigmoid)
        logits_scale_128_upsampled_to_256_sigmoid = tf.nn.sigmoid(logits_scale_128_upsampled_to_256,'scale_128_map')
        tf.summary.image('scale_128_map', logits_scale_128_upsampled_to_256_sigmoid)
        logits_scale_256_upsampled_to_256_sigmoid = tf.nn.sigmoid(logits_scale_256,'scale_256_map')
        tf.summary.image('scale_256_map', logits_scale_256_upsampled_to_256_sigmoid)

        logits_concat = tf.concat((logits_scale_64_3_upsampled_to_256,
                                   logits_scale_64_2_upsampled_to_256,
                                   logits_scale_64_1_upsampled_to_256,
                                   logits_scale_128_upsampled_to_256,
                                   logits_scale_256
                                   ), 3)
        logits = conv2d(logits_concat, 5, 1, 3, True)


        yp= tf.nn.sigmoid(logits,name="yp")
        tf.summary.image("yp", yp, 100)


        loss_64_3 =  fused_loss(logits_scale_64_3_upsampled_to_256_sigmoid,ysc)
        loss_64_2 =  fused_loss(logits_scale_64_2_upsampled_to_256_sigmoid,ysc)
        loss_64_1 =  fused_loss(logits_scale_64_1_upsampled_to_256_sigmoid,ysc)

        loss_128 = fused_loss(logits_scale_128_upsampled_to_256_sigmoid, ysc)
        loss_256 = fused_loss(logits_scale_256_upsampled_to_256_sigmoid,ysc)

        loss_yp = fused_loss(yp, ysc)

        cross_entropy = loss_yp+ loss_64_3 + loss_64_2 + loss_64_1+loss_128 + loss_256
        tf.summary.scalar("cross_entropy", cross_entropy)

        #measure

        MAE = tf.reduce_mean(tf.abs(yp - ysc),name="mae")
        tf.summary.scalar("MAE error", MAE)

        prec,recall,F_score = F_measure(ysc,yp)

        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        weight_decay = 0.0001

        train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(cross_entropy + weight_decay*l2 )

        #train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(MAE)

        #train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(MAE + l2 * weight_decay)

        #train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(cross_entropy + l2 * weight_decay)


    with tf.Session(graph=graph) as session:
        batch_size = 4
        learning_rate = 0.1/50
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=300)
        saver.restore(session, 'model_v61/dense_seg_v61_170.ckpt')

        merged_summary_op = tf.summary.merge_all()
        if tf.gfile.Exists(log_dir):
            tf.gfile.DeleteRecursively(log_dir)
        tf.gfile.MakeDirs(log_dir)
        test_summary_writer = tf.summary.FileWriter(log_dir + "/test")
        train_summary_writer = tf.summary.FileWriter(log_dir + "/train")


        for epoch in range(1, 1 + 30):
            if epoch == 5: learning_rate = 0.01/50
            if epoch == 10: learning_rate = 0.001/50

            #randmn data
            train_data, train_labels = data['train_data'], data['train_labels']
            pi = np.random.permutation(len(train_data))
            train_data, train_labels = train_data[pi], train_labels[pi]
            batch_count = len(train_data) / batch_size
            batches_data = np.split(train_data[:batch_count * batch_size], batch_count)
            batches_labels = np.split(train_labels[:batch_count * batch_size], batch_count)
            print("Batch per epoch: ", batch_count)

            for batch_idx in range(batch_count):

                xs_, ys_ = images_preprocessing(batches_data[batch_idx], batches_labels[batch_idx])

                batch_res = session.run([merged_summary_op, train_step, cross_entropy, MAE,prec,recall,F_score],
                                        feed_dict={xs: xs_, ys: ys_, lr: learning_rate, is_training: True,keep_prob: 0.8})
                if batch_idx % 100 == 0:
                    train_summary_writer.add_summary(batch_res[0], epoch * batch_count + batch_idx)
                    print(epoch, batch_idx, batch_res[2:])

            test_results = run_in_batch_avg(session, [merged_summary_op, cross_entropy, MAE,prec,recall,F_score], [xs, ys],
                                            test_summary_writer=test_summary_writer, epoch=epoch,
                                            feed_dict={xs: data['test_data'], ys: data['test_labels'],
                                                       is_training: False, keep_prob: 1.})

            save_path = saver.save(session, save_path='model_v61_v9/dense_seg_v61_%d.ckpt' % epoch)

            print(epoch, batch_res[2:], test_results)




def run():
    train_images = np.load("train_images.npy",mmap_mode='r')
    train_masks = np.load("train_masks.npy",mmap_mode='r')
    test_images = np.load("test_images.npy",mmap_mode='r')
    test_masks = np.load("test_masks.npy",mmap_mode='r')

    image_size = cg.image_size
    image_dim = image_size * image_size * 3
    label_count = image_size * image_size

    train_data, train_labels = train_images, train_masks
    test_data, test_labels = test_images, test_masks

    print ("Train:", np.shape(train_data), np.shape(train_labels))
    print ("Test:", np.shape(test_data), np.shape(test_labels))
    data = {'train_data': train_data,
            'train_labels': train_labels,
            'test_data': test_data,
            'test_labels': test_labels}
    run_model(data, image_dim, label_count)


run()