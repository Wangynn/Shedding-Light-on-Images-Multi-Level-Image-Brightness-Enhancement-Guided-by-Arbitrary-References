# coding: utf-8
from __future__ import print_function
import os, time, random
import tensorflow as tf
from PIL import Image
import numpy as np
from utils import *
from model import *
from glob import glob
import cv2

batch_size = 8
change_patch_size = 100

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sess = tf.Session()

input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')

# output of encoder net
[conv1, conv2, conv3, conv4, feature1] = EncoderNet(input_low)
[conv11, conv22, conv33, conv44, feature2] = EncoderNet(input_high)

# create a new feature
[low_feature_light, low_feature_content] = cut_feature(feature1)
[high_feature_light, high_feature_content] = cut_feature(feature2)

feature_new = match_feature(high_feature_light, low_feature_content)

# output of decoder net
output_high = DecoderNet(conv1, conv2, conv3, conv4, feature_new)

# the second pass through the encoder net
[_, _, _, _, feature_output] = EncoderNet(output_high)

[output_feature_light, output_feature_content] = cut_feature(feature_output)

# computer ssim
ssim = compute_ssim(input_high, output_high)

# define loss
def restoration_loss(input_high, output_high):
    loss_restoration = tf.reduce_mean(tf.abs(input_high - output_high))
    # loss_restoration = tf.reduce_sum(tf.losses.huber_loss(input_high, output_high))
    return loss_restoration

def feature_loss(low_content, output_content, low_light, high_light, output_light):
    low_light_norm = instance_normalization(low_light)
    high_light_norm = instance_normalization(high_light)
    output_light_norm = instance_normalization(output_light)
    d_ap = tf.reduce_mean(tf.square(high_light_norm - output_light_norm))
    d_an = tf.reduce_mean(tf.square(low_light_norm - output_light_norm))
    loss_light_feature = tf.reduce_max([d_ap-d_an+0.08, tf.constant(0, dtype=tf.float32)])
    loss_content_feature = tf.reduce_mean(tf.square(low_content - output_content))
    return loss_light_feature + loss_content_feature

# def content_loss(input_low, input_high, output_high):
#     input_low_hsv = rgb2hsv(input_low)
#     input_high_hsv = rgb2hsv(input_high)
#     output_high_hsv = rgb2hsv(output_high)
#     input_h,input_s,input_v=tf.split(input_low_hsv,[1,1,1],axis=3)
#     h,s,v=tf.split(input_low_hsv,[1,1,1],axis=3)
#     output_h,output_s,output_v=tf.split(output_high_hsv,[1,1,1],axis=3)
#     loss_h = tf.reduce_mean(tf.abs(input_h - output_h))
#     loss_s = tf.reduce_mean(tf.abs(input_s - output_s))
#     loss_v = tf.reduce_max([tf.reduce_mean(tf.abs(v - output_v))-tf.reduce_mean(tf.abs(input_v - output_v))+0.5,tf.constant(0, dtype=tf.float32)])
#     return (loss_h + loss_s + loss_v)/3.
def content_loss(input_low, output_high):
    input_low_hsv = rgb2hsv(input_low)
    output_high_hsv = rgb2hsv(output_high)
    input_h,input_s,input_v=tf.split(input_low_hsv,[1,1,1],axis=3)
    output_h,output_s,output_v=tf.split(output_high_hsv,[1,1,1],axis=3)
    input_h = tf.reshape(input_h, [batch_size,-1])
    output_h = tf.reshape(output_h, [batch_size,-1])
    input_s = tf.reshape(input_s, [batch_size,-1])
    output_s = tf.reshape(output_s, [batch_size,-1])
    norm_input_h = tf.nn.l2_normalize(input_h,axis=-1)        
    norm_output_h = tf.nn.l2_normalize(output_h,axis=-1)
    norm_input_s = tf.nn.l2_normalize(input_s,axis=-1)        
    norm_output_s = tf.nn.l2_normalize(output_s,axis=-1)
    loss_h = tf.reduce_mean(tf.losses.cosine_distance(norm_input_h, norm_output_h, axis=-1, reduction=tf.losses.Reduction.NONE))
    loss_s = tf.reduce_mean(tf.losses.cosine_distance(norm_input_s, norm_output_s, axis=-1, reduction=tf.losses.Reduction.NONE))
    # loss_h = tf.reduce_mean(tf.abs(input_h - output_h))
    # loss_s = tf.reduce_mean(tf.abs(input_s - output_s))
    return (loss_h + loss_s)/2.

loss_restoration = restoration_loss(input_high, output_high)
loss_feature = feature_loss(low_feature_content, output_feature_content, low_feature_light, high_feature_light, output_feature_light)
loss_content = content_loss(input_low, output_high)

train_loss = loss_restoration + 2*loss_feature + 2*loss_content

###initialize
lr = tf.placeholder(tf.float32, name='learning_rate')

optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')
var_train = [var for var in tf.trainable_variables() if 'EncoderNet' or 'DecoderNet' in var.name]  #需要进行训练更新的变量

train_op = optimizer.minimize(train_loss, var_list = var_train)
sess.run(tf.global_variables_initializer())

saver_train = tf.train.Saver(var_list = var_train)
print("[*] Initialize model successfully...")

#load data
###train_data
train_low_data = []
train_high_data = []
train_low_data_names = glob('./lol/train/low/*.png')  #输出文件list：('…/low/1.png'，'…/low/2.png'，…)
train_low_data_names.sort()
train_high_data_names = glob('./lol/train/high/*.png') 
train_high_data_names.sort()
assert len(train_low_data_names) == len(train_high_data_names)  #assert(condition)：条件成立才继续运行程序，不成立则终止报错
print('[*] Number of training data: %d' % len(train_low_data_names))
for idx in range(len(train_low_data_names)):
    low_im = load_images(train_low_data_names[idx])
    train_low_data.append(low_im)

    high_im = load_images(train_high_data_names[idx])
    train_high_data.append(high_im)


###eval_data
eval_low_data = []
eval_high_data = []
eval_low_data_names = glob('./lol/test/low/*.png')
eval_low_data_names.sort()
eval_high_data_names = glob('./lol/test/high/*.png')
eval_high_data_names.sort()
assert len(eval_low_data_names) == len(eval_high_data_names)
for idx in range(len(eval_low_data_names)):
    eval_low_im = load_images(eval_low_data_names[idx])
    eval_low_data.append(eval_low_im)

    eval_high_im = load_images(eval_high_data_names[idx])
    eval_high_data.append(eval_high_im)

epoch = 1000
learning_rate = 0.0001

sample_dir = './lol_nocut_96cos_changeda2/'
if not os.path.isdir(sample_dir):  #判断某一对象是否是目录（需要输入绝对路径）
    os.makedirs(sample_dir)  #递归创建目录

eval_every_epoch = 100
numBatch = len(train_low_data) // int(batch_size)

saver = saver_train

checkpoint_dir = './checkpoint/lol_nocut_96cos_changeda2/'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

start_step = 0
start_epoch = 0
iter_num = 0
print("[*] Start training with start epoch %d start iter %d : " % (start_epoch, iter_num))

start_time = time.time()
image_id = 0
psnr_for_save_epoch = []
ssim_for_save_epoch = []
mean_l_save_epoch = []
mean_c_save_epoch = []
var_l_save_epoch = []
var_c_save_epoch = []
mse_l_save_epoch = []
mse_c_save_epoch = []
for epoch in range(start_epoch, epoch):  #循环次数为总epoch数
    for batch_id in range(start_step, numBatch):  #循环次数为一个epoch中的batch个数
        batch_input_low = np.zeros((batch_size, 400, 600, 3), dtype="float32")   #一个batch的低光输入
        batch_input_high = np.zeros((batch_size, 400, 600, 3), dtype="float32") 
        change_patch = np.zeros((change_patch_size, change_patch_size, 3), dtype="float32")
    
        #数据增强更新batch_input_low/high
        for patch_id in range(batch_size):   #循环次数为一个batch中的patch个数
            h, w, _ = train_low_data[image_id].shape
            x = random.randint(0, h - change_patch_size)
            y = random.randint(0, w - change_patch_size)
            z = random.uniform(0, 1) 
            rand_mode = random.randint(0, 3)
            batch_input_low[patch_id, :, :, :] = augmentation(train_low_data[image_id], rand_mode)   #数据增强
            batch_input_high[patch_id, :, :, :] = augmentation(train_high_data[image_id], rand_mode)
            change_patch = z*batch_input_low[patch_id, x : x+change_patch_size, y : y+change_patch_size, :] + (1-z)*batch_input_high[patch_id, x : x+change_patch_size, y : y+change_patch_size, :]
            batch_input_low[patch_id, x : x+change_patch_size, y : y+change_patch_size, :] = change_patch

            image_id = (image_id + 1) % len(train_low_data)

            #每经过一个epoch，将训练数据重新排序，以在下一个epoch时获得不同batch
            if image_id == 0:  
                tmp = list(zip(train_low_data, train_high_data)) 
                random.shuffle(tmp)  
                train_low_data, train_high_data  = zip(*tmp)

        #对于每一个batch反向更新，计算损失并输出
        result = sess.run(output_high, feed_dict={input_low: batch_input_low, input_high: batch_input_high})
        train_psnr = np.mean(compute_psnr(result, batch_input_high))

        _,  restoration_loss, content_loss, loss = sess.run([train_op, loss_restoration, loss_content, train_loss], feed_dict={input_low: batch_input_low, input_high: batch_input_high, lr: learning_rate})

        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, restoration_loss: %.6f, content_loss: %.6f, loss: %.6f, psnr: %.6f "\
                % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, restoration_loss, content_loss, loss, train_psnr))
        iter_num += 1

    #每100个epoch保存eval_data经过decomposition_net分解后的结果图像
    eval_img_name = []
    if (epoch + 1) % eval_every_epoch == 0:
        print("[*] Evaluating for epoch %d..." % (epoch + 1))
        psnr_for_eval_epoch = []
        ssim_for_eval_epoch = []
        mean_l_eval_epoch = []
        mean_c_eval_epoch = []
        var_l_eval_epoch = []
        var_c_eval_epoch = []
        mse_l_eval_epoch = []
        mse_c_eval_epoch = []
        for idx in range(len(eval_low_data_names)):
            [_, name] = os.path.split(eval_low_data_names[idx])  #返回文件的路径和文件名
            suffix = name[name.find('.') + 1:]  
            name = name[:name.find('.')]
            eval_img_name.append(name)
            name = eval_img_name[idx]
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)
            input_high_eval = np.expand_dims(eval_high_data[idx], axis=0)
            eval_ssim, light_1, content_1, light_2, content_2, output = sess.run([ssim, low_feature_light, low_feature_content, high_feature_light, high_feature_content, output_high], feed_dict={input_low: input_low_eval, input_high: input_high_eval})
            eval_psnr = compute_psnr(output, input_high_eval)
            mean_l, var_l, mse_l = computer_error(light_1, light_2)
            mean_c, var_c, mse_c = computer_error(content_1, content_2)
            save_images(os.path.join(sample_dir, '%s_%d-psnr:%.6f.png' % ( name, epoch + 1, eval_psnr)),  output)
            psnr_for_eval_epoch.append(eval_psnr)
            ssim_for_eval_epoch.append(eval_ssim)
            mean_l_eval_epoch.append(mean_l)
            mean_c_eval_epoch.append(mean_c)
            var_l_eval_epoch.append(var_l)
            var_c_eval_epoch.append(var_c)
            mse_l_eval_epoch.append(mse_l)
            mse_c_eval_epoch.append(mse_c)

        psnr_for_save_epoch.append(np.mean(psnr_for_eval_epoch))
        ssim_for_save_epoch.append(np.mean(ssim_for_eval_epoch))
        mean_l_save_epoch.append(np.mean(mean_l_eval_epoch))
        mean_c_save_epoch.append(np.mean(mean_c_eval_epoch))
        var_l_save_epoch.append(np.mean(var_l_eval_epoch))
        var_c_save_epoch.append(np.mean(var_c_eval_epoch))
        mse_l_save_epoch.append(np.mean(mse_l_eval_epoch))
        mse_c_save_epoch.append(np.mean(mse_c_eval_epoch))

    saver.save(sess, checkpoint_dir + 'model.ckpt')


print("The eval pictures:")
number = 0
for idx in range(len(psnr_for_save_epoch)):
    number = (idx+1)*eval_every_epoch
    print("[*] The average psnr of %d epoch is %.6f." % (number, psnr_for_save_epoch[idx]))
    print("[*] The average ssim of %d epoch is %.6f." % (number, ssim_for_save_epoch[idx]))
    print("[*] The mean/var/mse of %d epoch: (0-95) %.6f, %.6f, %.6f,   (96-511) %.6f, %.6f, %.6f." % (number, mean_l_save_epoch[idx], var_l_save_epoch[idx], mse_l_save_epoch[idx], mean_c_save_epoch[idx], var_c_save_epoch[idx], mse_c_save_epoch[idx]))

print("[*] Finish training.")
