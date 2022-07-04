import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import nn

def lrelu(x, trainbable=None):
    return tf.maximum(x*0.2,x)

def upsample_and_concat(x1, x2, output_channels, in_channels, scope_name, trainable=True):
    #先进行一次deconvolution，再concatenate
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool_size = 2
        deconv_filter = tf.get_variable('weights', [pool_size, pool_size, output_channels, in_channels], trainable= True)  #tf.get_variable(name,shape,dtype,…) \output/in_channels没反？？
                                                                                                                           #tf.get_variable和tf.Variable默认trainable=True
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1], name=scope_name)

        deconv_output =  tf.concat([deconv, x2],3)  #tf.concat([tensor1,tenssor2,…],axis) \此处3指通道维度
        deconv_output.set_shape([None, None, None, output_channels*2])  #set_shape()用于更新已有的某个tensor的shape，tf.reshape()用于创建新的tensor

        return deconv_output

def EncoderNet(input):
    with tf.variable_scope('EncoderNet', reuse=tf.AUTO_REUSE):
        conv1=slim.conv2d(input,32,[3,3], rate=1, activation_fn=lrelu,scope='en_conv1_1')
        conv1=slim.conv2d(conv1,32,[3,3], rate=1, activation_fn=lrelu,scope='en_conv1_2')
        pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )

        conv2=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=lrelu,scope='en_conv2_1')
        conv2=slim.conv2d(conv2,64,[3,3], rate=1, activation_fn=lrelu,scope='en_conv2_2')
        pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )

        conv3=slim.conv2d(pool2,128,[3,3], rate=1, activation_fn=lrelu,scope='en_conv3_1')
        conv3=slim.conv2d(conv3,128,[3,3], rate=1, activation_fn=lrelu,scope='en_conv3_2')
        pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )

        conv4=slim.conv2d(pool3,256,[3,3], rate=1, activation_fn=lrelu,scope='en_conv4_1')
        conv4=slim.conv2d(conv4,256,[3,3], rate=1, activation_fn=lrelu,scope='en_conv4_2')
        pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )

        conv5=slim.conv2d(pool4,512,[3,3], rate=1, activation_fn=lrelu,scope='en_conv5_1')
        # conv5=slim.conv2d(conv5,512,[3,3], rate=1, activation_fn=lrelu,scope='de_conv5_2')

        feature = tf.reduce_mean(conv5,axis=[1,2])
        
        return conv1, conv2, conv3, conv4, feature


def DecoderNet(conv1, conv2, conv3, conv4, feature):
    with tf.variable_scope('DecoderNet', reuse=tf.AUTO_REUSE):
        conv_dense = tf.layers.dense(feature,units=256,activation=tf.nn.relu)
        feature = tf.expand_dims(conv_dense,axis=1)
        feature = tf.expand_dims(feature,axis=2)
        ones = tf.zeros(shape=tf.shape(conv4))
        global_feature = feature + ones

        up6 =  tf.concat([conv4, global_feature], axis=3, name='up_6')
        conv6=slim.conv2d(up6,  256,[3,3], rate=1, activation_fn=lrelu,scope='en_conv6_1')
        conv6=slim.conv2d(conv6,256,[3,3], rate=1, activation_fn=lrelu,scope='en_conv6_2')

        up7 =  upsample_and_concat( conv6, conv3, 128, 256, 'up_7'  )
        conv7=slim.conv2d(up7,  128,[3,3], rate=1, activation_fn=lrelu,scope='en_conv7_1')
        conv7=slim.conv2d(conv7,128,[3,3], rate=1, activation_fn=lrelu,scope='en_conv7_2')

        up8 =  upsample_and_concat( conv7, conv2, 64, 128, 'up_8' )
        conv8=slim.conv2d(up8,  64,[3,3], rate=1, activation_fn=lrelu,scope='en_conv8_1')
        conv8=slim.conv2d(conv8,64,[3,3], rate=1, activation_fn=lrelu,scope='en_conv8_2')

        up9 =  upsample_and_concat( conv8, conv1, 32, 64, 'up_9' )
        conv9=slim.conv2d(up9,  32,[3,3], rate=1, activation_fn=lrelu,scope='en_conv9_1')
        conv9=slim.conv2d(conv9,32,[3,3], rate=1, activation_fn=lrelu,scope='en_conv9_2')

        conv10=slim.conv2d(conv9,3,[1,1], rate=1, activation_fn=None, scope='en_conv10')
        out = tf.sigmoid(conv10)

        # conv9 = input1 * conv9
        # deconv_filter = tf.Variable(tf.truncated_normal([2, 2, 3, 16], stddev=0.02))
        # conv10 = tf.nn.conv2d_transpose(conv9, deconv_filter, tf.shape(input), strides=[1, 2, 2, 1])
        # out = slim.conv2d(conv10, 3, [3, 3],rate=1,activation_fn=nn.tanh,scope='out') * 0.58 + 0.52

        return out
