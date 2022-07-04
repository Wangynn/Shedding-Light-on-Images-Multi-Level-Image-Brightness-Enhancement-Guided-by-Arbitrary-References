## 参考图像为多张时，对结果图像取平均

# coding: utf-8
from __future__ import print_function
import os
import time
import random
from PIL import Image
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from glob import glob
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sess = tf.Session()

input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')

[conv1, conv2, conv3, conv4, feature1] = EncoderNet(input_low)
[conv11, conv22, conv33, conv44, feature2] = EncoderNet(input_high)

[low_feature_light, low_feature_content] = cut_feature(feature1)
[high_feature_light, high_feature_content] = cut_feature(feature2)

feature_new = match_feature(high_feature_light, low_feature_content)

output_high = DecoderNet(conv1, conv2, conv3, conv4, feature_new)

var_train = [var for var in tf.trainable_variables() if 'EncoderNet' or 'DecoderNet' in var.name]

saver_train = tf.train.Saver(var_list = var_train)


train_checkpoint_dir ='./checkpoint/lol_nocut_96cos_changeda/'
ckpt_pre=tf.train.get_checkpoint_state(train_checkpoint_dir)
if ckpt_pre:
    print('loaded '+ckpt_pre.model_checkpoint_path)
    saver_train.restore(sess,ckpt_pre.model_checkpoint_path)
else:
    print('No train net checkpoint!')


###load eval data
eval_low_data = []
eval_img_name =[]
eval_low_data_name = glob('./user_study/test_pic/*.*')
eval_low_data_name.sort()
for idx in range(len(eval_low_data_name)):
    [_, name] = os.path.split(eval_low_data_name[idx])  #返回文件的路径和文件名
    suffix = name[name.find('.') + 1:]  
    name = name[:name.find('.')]
    eval_img_name.append(name)
    eval_low_im = load_images(eval_low_data_name[idx])
    # img = Image.open(eval_low_data_name[idx])
    # new_img = img.resize((600,400),Image.BILINEAR)
    # eval_low_im = np.array(new_img, dtype="float32") / 255.0
    eval_low_data.append(eval_low_im)

# eval_high_data = []
# eval_high_data_name = glob('./lol/test/high/*.png')
# eval_high_data_name.sort()
# for idx in range(len(eval_high_data_name)):
#     eval_high_im = load_images(eval_high_data_name[idx])
#     # img = Image.open(eval_high_data_name[idx])
#     # new_img = img.resize((600,400),Image.BILINEAR)
#     # eval_high_im = np.array(new_img, dtype="float32") / 255.0
#     eval_high_data.append(eval_high_im)


print("[*] Start evalating!")
start_time = time.time()


# #### 计算第1000个epoch的psnr、ssim，特征向量的mean、var、mse
# assert len(eval_low_data_name) == len(eval_high_data_name)
# print("[*] The number of eval pictures are %d." % len(eval_high_data_name))
# psnr_for_one_picture = []
# ssim_for_one_picture = []
# mean_l_one_picture = []
# mean_c_one_picture = []
# var_l_one_picture = []
# var_c_one_picture = []
# mse_l_one_picture = []
# mse_c_one_picture = []

# for idx in range(len(eval_low_data)):
#     input_high_eval = eval_high_data[idx]
#     input_high_eval = np.expand_dims(input_high_eval, axis=0)

#     input_low_eval = eval_low_data[idx]
#     input_low_eval = np.expand_dims(input_low_eval, axis=0)

#     light_1, content_1, light_2, content_2, output = sess.run([low_feature_light, low_feature_content, high_feature_light, high_feature_content, output_high], feed_dict={input_low: input_low_eval, input_high: input_high_eval})
#     eval_psnr = compute_psnr(output, input_high_eval)
#     eval_ssim = compute_ssim(output, input_high_eval)
#     mean_l, var_l, mse_l = computer_error(light_1, light_2)
#     mean_c, var_c, mse_c = computer_error(content_1, content_2)
#     psnr_for_one_picture.append(eval_psnr)
#     ssim_for_one_picture.append(eval_ssim)
#     mean_l_one_picture.append(mean_l)
#     mean_c_one_picture.append(mean_c)
#     var_l_one_picture.append(var_l)
#     var_c_one_picture.append(var_c)
#     mse_l_one_picture.append(mse_l)
#     mse_c_one_picture.append(mse_c)

# print("[*] The average psnr of 1000 epoch is %.6f." % (np.mean(psnr_for_one_picture)))
# print("[*] The average ssim of 1000 epoch is %.6f." % (np.mean(ssim_for_one_picture)))
# print("[*] The mean/var/mse of 1000 epoch: (0-95) %.6f, %.6f, %.6f,   (96-511) %.6f, %.6f, %.6f." % (np.mean(mean_l_one_picture), np.mean(var_l_one_picture), np.mean(mse_l_one_picture), np.mean(mean_c_one_picture), np.mean(var_c_one_picture), np.mean(mse_c_one_picture)))



#### 生成图像
sample_dir = './user_study/results/3_lihaotian/'
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)

# ### 参考图像与暗图一一对应，并计算psnr
# assert len(eval_low_data_name) == len(eval_high_data_name)
# psnr_for_pictures = []
# ssim_for_pictures = []
# for idx in range(len(eval_low_data)):
#     input_high_eval = eval_high_data[idx]
#     input_high_eval = np.expand_dims(input_high_eval, axis=0)

#     name = eval_img_name[idx]
#     input_low_eval = eval_low_data[idx]
#     input_low_eval = np.expand_dims(input_low_eval, axis=0)

#     output = sess.run(output_high, feed_dict={input_low: input_low_eval, input_high: input_high_eval})
#     save_images(os.path.join(sample_dir, '%s.png' % (name)), output)
#     eval_psnr = compute_psnr(output, input_high_eval)
#     eval_ssim = compute_ssim(output, input_high_eval)
#     psnr_for_pictures.append(eval_psnr)
#     ssim_for_pictures.append(eval_ssim)
# print("[*] The average psnr of 1000 epoch is %.6f." % (np.mean(psnr_for_pictures)))
# print("[*] The average ssim of 1000 epoch is %.6f." % (np.mean(ssim_for_pictures)))


### 一张参考图像转换多张暗图
eval_high_data = []
eval_high_im = load_images('./user_study/reference/14.png')
eval_high_data.append(eval_high_im)
if len(eval_high_data)==1:
    print("[*] The number of reference picture is one.")
    input_high_eval = eval_high_data[0]
    input_high_eval = np.expand_dims(input_high_eval, axis=0)
    for idx in range(len(eval_low_data)):
        name = eval_img_name[idx]
        input_low_eval = eval_low_data[idx]
        input_low_eval = np.expand_dims(input_low_eval, axis=0)

        output = sess.run(output_high, feed_dict={input_low: input_low_eval, input_high: input_high_eval})
        save_images(os.path.join(sample_dir, '%s.png' % (name)), output)
elif len(eval_high_data)>1:
    print("[*] Please input one reference picture.")


print("[*] Finish testing.")


    
