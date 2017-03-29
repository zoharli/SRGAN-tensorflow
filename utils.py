import tensorflow as tf

def conv_layer(input,filter,stride,name='conv'):
    with tf.variable_scope(name):
        conv=tf.nn.conv2d(input,filter,strides=[1,stride,stride,1],padding='SAME')
        bias=tf.Variable(tf.zeros(conv.shape.as_list()[3]),name='bias')
        return tf.nn.bias_add(conv,bias)
def batch_norm(input,name='BN'):
    ch_num=input.shape.as_list()[3]
    return tf.nn.fused_batch_norm(input,
                                  offset=tf.Variable(tf.zeros(ch_num),name='offset'),
                                  scale=tf.Variable(tf.ones(ch_num),name='scale'),
                                  name=name)[0]
def conv_filter(shape,name='filter'):
    return tf.Variable(tf.truncated_normal(shape),name=name)
def leaky_relu(x,alpha=0.1,name='lrelu'):
     with tf.name_scope(name):
         x=tf.maximum(x,alpha*x)
         return x

