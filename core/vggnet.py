from scipy.io import loadmat
import tensorflow as tf

vgg_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
          'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
          'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
          'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
          'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4']

class Vgg(object):
    def __init__(self, path):
        self.path = path
        self.params = {}
        self.images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'images')
        self.features = tf.placeholder(tf.float32, [None, 196, 512], 'features')

    def conv_same(self, inputs, w, b):
        return tf.add(tf.nn.conv2d(inputs, w, strides=[1,1,1,1], padding='SAME'), b)

    def relu(self, inputs):
        return tf.nn.relu(inputs)

    def maxpooling(self, inputs):
        return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    def get_variable(self, name_layer, weights_type, weights):
        name = name_layer + '/' + weights_type
        return tf.get_variable(name, initializer=tf.constant(weights))
            
    def build(self):
        model = loadmat(self.path)
        layers = model['layers'][0]
        with tf.variable_scope('encoder'):
            for layer in layers:
                type_layer = layer[0][0][1][0]
                name_layer = layer[0][0][0][0]
                if type_layer == 'conv':
                    w = layer[0][0][2][0][0].transpose(1, 0, 2, 3)
                    b = layer[0][0][2][0][1].reshape(-1)
                    self.params[name_layer] = {}
                    self.params[name_layer]['w'] = self.get_variable(name_layer, 'w', w)
                    self.params[name_layer]['b'] = self.get_variable(name_layer, 'b', b)

        inputs = self.images
        for layer in vgg_layers:
            type_layer = layer[: 4]
            if type_layer == 'relu':
                inputs = self.relu(inputs)
            elif type_layer == 'pool':
                inputs = self.maxpooling(inputs)
            elif type_layer == 'conv':
                inputs = self.conv_same(inputs, self.params[layer]['w'], self.params[layer]['b'])
            if layer == 'conv5_3':
                self.features = tf.reshape(inputs, [-1, 196, 512])