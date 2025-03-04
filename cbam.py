import tensorflow as tf
from tensorflow.keras import layers, models

# Channel Attention Module (CBAM) in TensorFlow/Keras
class CBAM(layers.Layer):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel Attention
        self.channel_avg_pool = layers.GlobalAveragePooling2D()
        self.channel_conv1 = layers.Conv2D(channels // reduction, kernel_size=1, use_bias=False)
        self.channel_relu = layers.ReLU()
        self.channel_conv2 = layers.Conv2D(channels, kernel_size=1, use_bias=False)
        self.channel_sigmoid = layers.Activation('sigmoid')
        
        # Spatial Attention
        self.spatial_conv = layers.Conv2D(1, kernel_size=7, padding='same', use_bias=False)
        self.spatial_sigmoid = layers.Activation('sigmoid')
        
    def call(self, inputs):
        # Channel Attention
        avg_pool = self.channel_avg_pool(inputs)
        avg_pool = tf.expand_dims(tf.expand_dims(avg_pool, 1), 1)  # Shape: (batch_size, 1, 1, channels)
        channel_attention = self.channel_conv1(avg_pool)
        channel_attention = self.channel_relu(channel_attention)
        channel_attention = self.channel_conv2(channel_attention)
        channel_attention = self.channel_sigmoid(channel_attention)
        
        x = inputs * channel_attention
        
        # Spatial Attention
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        spatial_input = tf.concat([max_pool, avg_pool], axis=-1)
        spatial_attention = self.spatial_conv(spatial_input)
        spatial_attention = self.spatial_sigmoid(spatial_attention)
        
        x = x * spatial_attention
        
        return x
