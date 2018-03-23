from __future__ import division
import tensorflow as tf
from utils import *

import tensorflow as tf
import tensorflow.contrib.slim as slim

def instance_norm(x, scope_name = "instance_norm"):
    with tf.variable_scope(scope_name):
        channel = x.get_shape()[3]
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale',[channel], initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset',[channel],initializer=tf.constant_initializer(0.0))
        out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset
        return out

def lrelu(x, leak=0.2, scope_name="lrelu"):
     with tf.variable_scope(scope_name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

def ResnetBlock(x, dim, use_dropout, name="resnetblock"):
    """
    so far, only padding_type=reflect is supported for tf.pad
    and padding="VALID" is supported for slim.conv2d
    """
    with tf.variable_scope(name):
        output = tf.pad(x,[[0,0],[1,1],[1,1],[0,0]],"REFLECT") 
        output = slim.conv2d(output, num_outputs=dim, kernel_size=[3, 3], stride=1, padding="VALID",
                             normalizer_fn=None, activation_fn = None, scope=name+"_conv1")
        output = instance_norm(output, scope_name = name + "_instance_norm1")
        output = tf.nn.relu(output) 

        if use_dropout:
            output = slim.dropout(output, keep_prob=0.5, scope=name+"_dropout")

        output = tf.pad(output, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
        output = slim.conv2d(output, num_outputs=dim, kernel_size=[3, 3], stride=1, padding="VALID",
                            normalizer_fn=None, activation_fn=None, 
                            scope=name+"_conv2")
        output = instance_norm(output, scope_name = name + "_instance_norm2")
        return x + output

def ResnetGenerator(x, output_nc, ngf=64, use_dropout=False, n_blocks=6, name="ResnetGenerator_6"):
    """
    in original pytorch verion, bias is added to conv2d, while bias is ignored here
    """
    with tf.variable_scope(name):
        x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], "REFLECT")
        output = slim.conv2d(x, num_outputs=ngf, kernel_size=[7, 7], padding="VALID",
                           normalizer_fn=None, activation_fn=None, scope=name+"_conv")
        output = instance_norm(output, scope_name = name + "_instance_norm")
        output = tf.nn.relu(output) 

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            output = slim.conv2d(output, num_outputs=ngf * mult * 2, kernel_size=[3, 3],
                               stride=2, padding="SAME",
                               normalizer_fn=None, activation_fn=None, scope=name+"_conv_"+str(i))
            output = instance_norm(output, scope_name = name + "_instance_norm_" + str(i))
            output = tf.nn.relu(output)

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            output = ResnetBlock(output, ngf * mult,use_dropout=use_dropout, name=name+"_resnetblock_"+str(i)) 


        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            output = slim.conv2d_transpose(output, num_outputs=int(ngf*mult/2), kernel_size=[3, 3], stride=2,
                                            padding="SAME",normalizer_fn=None, activation_fn=None,
                                            scope=name+"_convtrans_"+str(i))
            output = instance_norm(output, scope_name = name + "_transpose_instance_norm_"+str(i))
            output = tf.nn.relu(output)

        output = tf.pad(output, [[0,0],[3,3],[3,3],[0,0]],"REFLECT") 
        output = slim.conv2d(output, num_outputs=output_nc, kernel_size=[7, 7], stride=1, padding="VALID",
                            normalizer_fn=None, activation_fn=None, scope=name+"_conv_last")

        output = tf.nn.tanh(output, name + "_tanh")

        return output


def NLayerDiscriminator(x, ndf=64, n_layers=3, use_sigmoid=False, name="Discriminator"):
    with tf.variable_scope(name):
        kw=4
        output = slim.conv2d(x, num_outputs=ndf, kernel_size=[kw, kw], stride=2, padding="SAME",
                            normalizer_fn=None, activation_fn=None, scope=name+"_conv") 
        output = lrelu(output, scope_name = name + "_leakyrelu1")

        nf_mult=1
        nf_mult_prev=1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            output = slim.conv2d(output, num_outputs=ndf * nf_mult, kernel_size=[kw, kw], stride=2,padding="SAME",
                                normalizer_fn=None, activation_fn=None, scope=name+"_conv_"+str(n)) 
            output = instance_norm(output, scope_name = name + "_first_instance_norm"+ str(n))
            output = tf.nn.relu(output, name + "_firstconv_lrelu"+str(n))

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)            
        output = slim.conv2d(output, num_outputs=ndf * nf_mult, kernel_size=[kw, kw], stride=1,
                            padding="SAME",
                            normalizer_fn=None, activation_fn=None, scope=name+"_second_conv_"+str(n_layers))
        output = instance_norm(output, scope_name = name + "_second_instance_norm")
        output = lrelu(output, scope_name = name + "_secondconv_lrelu")

        output = slim.conv2d(output, 1, kernel_size=[kw, kw], stride=1,padding="SAME",
                            normalizer_fn=None, activation_fn=None, scope=name+"_last_conv")

        if use_sigmoid:
            output = tf.nn.Sigmoid(output)

        return output 
            
def define_G(x, output_nc, ngf, n_blocks, use_dropout=False, reuse=False, scope_name = "d_G"):
    with tf.variable_scope(scope_name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        output = ResnetGenerator(x, output_nc=output_nc, ngf=ngf, use_dropout=use_dropout, n_blocks=n_blocks, name = scope_name + "_Generator")
        return output

def define_D(x, ndf, n_layers_D=3, use_sigmoid=False, reuse=False,scope_name = "d_D"):
    with tf.variable_scope(scope_name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        outputs = NLayerDiscriminator(x, ndf, n_layers_D, use_sigmoid=use_sigmoid, name = scope_name + "_Discriminator")
        return outputs

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
