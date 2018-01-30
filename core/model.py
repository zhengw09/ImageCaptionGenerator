#from __future__ import division
import tensorflow as tf

class CaptionGenerator(object):
    def __init__(self, word_to_idx, feature_dim=[196, 512], embed_dim=512, hidden_dim=1024, len_sent=16, lamda=0.0):
        self.word_to_idx = word_to_idx
        self.idx_to_word = {v : k for k, v in word_to_idx.items()}
        self.lamda = lamda
        self.K = len(word_to_idx)
        self.L = feature_dim[0]
        self.D = feature_dim[1]
        self.M = embed_dim
        self.N = hidden_dim
        self.T = len_sent

        #tf.contrib.layers.variance_scaling_initializer()
        #tf.contrib.layers.xavier_initializer()
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        #random_uniform_initializer(mean, stddev)
        #random_normal_initializer(minval, maxval)
        self.embedding_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
        self.feature = tf.placeholder(tf.float32, [None, self.L, self.D])
        self.caption = tf.placeholder(tf.int32, [None, self.T + 1])

    def get_variable(self, name, dim, initializer=None):
        return tf.get_variable(name, dim, initializer=initializer)

    def init_lstm(self, feature):
        with tf.variable_scope('initial_lstm'):
            feature_mean = tf.reduce_mean(feature, axis=1)

            h_w = self.get_variable('h_w', [self.D, self.N], self.weight_initializer)
            h_b = self.get_variable('h_b', [self.N], self.const_initializer)
            h = tf.nn.tanh(tf.matmul(feature_mean, h_w) + h_b)

            c_w = self.get_variable('c_w', [self.D, self.N], self.weight_initializer)
            c_b = self.get_variable('c_b', [self.N], self.const_initializer)
            c = tf.nn.tanh(tf.matmul(feature_mean, c_w) + c_b)
            return h, c  #batch_size * H

    def word_embedding(self, captions, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = self.get_variable('w', [self.K, self.M], self.embedding_initializer)
            return tf.nn.embedding_lookup(w, captions, name='vector_of_word') # (batch_size, T, M) or (batch_size, M)

    def projection_feature(self, feature):
        with tf.variable_scope('projection'):
            w_trans = self.get_variable('w', [self.D, self.D], self.weight_initializer)
            return tf.reshape(tf.matmul(tf.reshape(feature, [-1, self.D]), w_trans), [-1, self.L, self.D])

    def attention_layer(self, feature, projection, h, reuse=False):
        with tf.variable_scope('attention', reuse=reuse): #equation 4 and 5 and 9
            w = self.get_variable('w', [self.N, self.D], self.weight_initializer)
            b = self.get_variable('b', [self.D], self.const_initializer)
            w_att = self.get_variable('w_att', [self.D, 1], self.weight_initializer)
    
            h_att = tf.reshape(tf.nn.relu(projection + tf.expand_dims(tf.matmul(h, w), 1) + b), [-1, self.D])
            alpha = tf.nn.softmax(tf.reshape(tf.matmul(h_att, w_att), [-1, self.L]))
            context = tf.reduce_sum(feature * tf.expand_dims(alpha, -1), axis=1, name='context')
            return context, alpha #context batch_size * D

    def selector(self, context, h, reuse=False): #doubley stochastic attention
        with tf.variable_scope('selector', reuse=reuse):
            w = self.get_variable('w', [self.N, 1], self.weight_initializer)
            b = self.get_variable('b', [1], self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, name='beta') #batch_size * 1
            context = tf.multiply(context, beta, name='better_context')
            #context *= context * beta
            return context, beta

    def prediction(self, embedding, context, h, reuse=False):  #equation 7
        with tf.variable_scope('predict', reuse=reuse):
            h_w = self.get_variable('h_w', [self.N, self.M], self.weight_initializer)
            h_b = self.get_variable('h_b', [self.M], self.const_initializer)
            c_w = self.get_variable('c_w', [self.D, self.M], self.weight_initializer)
            w = self.get_variable('w', [self.M, self.K], self.weight_initializer)
            b = self.get_variable('b', [self.K], self.const_initializer)

            h = tf.nn.dropout(h, 0.5)
            intermidiate_output = tf.nn.tanh(tf.matmul(h, h_w) + tf.matmul(context, c_w) + h_b + embedding)
            intermidiate_output = tf.nn.dropout(intermidiate_output, 0.5)
            return tf.matmul(intermidiate_output, w) + b

    def loss_cal(self, loss, pred, captions, mask, lamda, alpha_list):
        loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=captions, logits=pred) * mask)
        loss += lamda * tf.reduce_sum((16.0 / 196 - tf.reduce_sum(tf.transpose(tf.stack(alpha_list), (1, 0, 2)), 1)) ** 2)
        return loss

    def batch_norm(self, inputs, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=inputs, decay=0.95, center=True, scale=True, is_training=(mode=='train'), updates_collections=None, scope=(name+'batch_norm'))

    def build_model(self):
        captions_in = self.caption[:, : self.T]
        captions_out = self.caption[:, 1 :]
        mask = tf.to_float(tf.not_equal(captions_out, self.word_to_idx['<NULL>']))

        loss = 0.0
        alpha_list = []
        cell_lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.N)
        #pre-processing data
        feature = self.feature
        feature = self.batch_norm(self.feature, mode='train', name='conv_features')
        h, c = self.init_lstm(feature)
        embedding = self.word_embedding(captions_in)
        projection = self.projection_feature(feature)

        for i in range(self.T):
            context, alpha = self.attention_layer(feature, projection, h, reuse=(i!=0))
            alpha_list.append(alpha)
            context, beta = self.selector(context, h, reuse=(i!=0))

            with tf.variable_scope('lstm', reuse=(i!=0)):
                _, (c, h) = cell_lstm(inputs=tf.concat([embedding[:, i, :], context], 1), state=[c, h])

            pred = self.prediction(embedding[:, i, :], context, h, reuse=(i!=0))

            loss += self.loss_cal(loss, pred, captions_out[:, i], mask[:, i], self.lamda, alpha_list)

        return loss / tf.to_float(tf.shape(feature)[0])

    def build_sampler(self, max_len=20):
        predict_result = []
        feature = self.feature
        feature = self.batch_norm(self.feature, mode='test', name='conv_features')
        h, c = self.init_lstm(feature)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.N)
        projection = self.projection_feature(feature)

        for i in range(max_len):
            if i != 0:
                x = self.word_embedding(sampled_word, reuse=True)
            else:
                x = self.word_embedding(tf.fill([tf.shape(feature)[0]], self.word_to_idx['<START>']))

            context, _  = self.attention_layer(feature, projection, h, reuse=(i!=0))
            context, _ = self.selector(context, h, reuse=(i!=0))

            with tf.variable_scope('lstm', reuse=(i!=0)):
                _, (c, h) = lstm_cell(tf.concat([x, context], 1), state=[c, h])

            pred = self.prediction(x, context, h, reuse=(i!=0))
            sampled_word = tf.argmax(pred, 1)
            predict_result.append(sampled_word)
 
        sampled_captions = tf.transpose(tf.stack(predict_result), (1, 0))     # (N, max_len)
        return sampled_captions
