import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os
import sys
import cPickle as pickle
from scipy import ndimage
from utils import *
import cPickle as pickle
sys.path.append('../coco-caption')
from pycocoevalcap.bleu.bleu import Bleu


class CaptioningSolver(object):
    def __init__(self, model, data, dev_data, **kwargs):

        self.model = model
        self.data = data
        self.dev_data = dev_data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 5)
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 1)
        self.model_path = kwargs.pop('model_path', './model/')
        self.test_model = kwargs.pop('test_model', None)
        self.optimizer = tf.train.AdamOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)


    def train(self):
        # train/dev dataset
        n_examples = self.data['captions'].shape[0]
        n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))
        features = self.data['features']
        captions = self.data['captions']
        image_idxs = self.data['image_idxs']
        dev_features = self.dev_data['features']
        n_iters_dev = int(np.ceil(float(dev_features.shape[0])/self.batch_size))

        # build graphs for training model and sampling captions
        loss = self.model.build_model()
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            tf.get_variable_scope().reuse_variables()
            generated_captions = self.model.build_sampler()

        # train op
        with tf.name_scope('optimizer'):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
        

        print "\nThe number of epoch: %d" %self.n_epochs
        print "Data size: %d" %n_examples
        print "Batch size: %d" %self.batch_size
        print "Iterations per epoch: %d" %n_iters_per_epoch
        
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            saver = tf.train.Saver(max_to_keep=40)

            prev_loss = -1
            curr_loss = 0
            start_t = time.time()

            for e in range(self.n_epochs):
                rand_idxs = np.random.permutation(n_examples)
                captions = captions[rand_idxs]
                image_idxs = image_idxs[rand_idxs]

                for i in range(n_iters_per_epoch):
                    captions_batch = captions[i*self.batch_size:(i+1)*self.batch_size]
                    image_idxs_batch = image_idxs[i*self.batch_size:(i+1)*self.batch_size]
                    features_batch = features[image_idxs_batch]
                    feed_dict = {self.model.feature: features_batch, self.model.caption: captions_batch}
                    _, l = sess.run([train_op, loss], feed_dict)
                    curr_loss += l

                    if (i+1) % self.print_every == 0:
                        print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" %(e+1, i+1, l)
                        ground_truths = captions[image_idxs == image_idxs_batch[0]]
                        decoded = decode_captions(ground_truths, self.model.idx_to_word)
                        for j, gt in enumerate(decoded):
                            print "Ground truth %d: %s" % (j+1, gt)                    
                        gen_caps = sess.run(generated_captions, feed_dict)
                        decoded = decode_captions(gen_caps, self.model.idx_to_word)
                        print "Generated caption: %s\n" % decoded[0]

                print "\nPrevious epoch loss: ", prev_loss
                print "Current epoch loss: ", curr_loss
                print "Elapsed time: ", time.time() - start_t
                prev_loss = curr_loss
                curr_loss = 0

                # calculate BLEU score on dev data
                dev_gen_cap = np.ndarray((dev_features.shape[0], 20))
                for i in range(n_iters_dev):
                    features_batch = dev_features[i*self.batch_size:(i+1)*self.batch_size]
                    feed_dict = {self.model.feature: features_batch}
                    gen_cap = sess.run(generated_captions, feed_dict=feed_dict)
                    dev_gen_cap[i*self.batch_size:(i+1)*self.batch_size] = gen_cap
                
                dev_decoded = decode_captions(dev_gen_cap, self.model.idx_to_word)
                save_pickle(dev_decoded, "./data/dev/dev.predicted.captions.pkl")
                print('Bleu_1 score on dev data: %s' % self.get_bleu(data_path='./data', split='dev'))
                           
                # save model's parameters
                if (e+1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e+1)
                    print "model-%s saved." %(e+1)
            
         
    def test(self, data, **kwargs):
        # build a graph to sample captions
        display = kwargs.pop('display', False)
        get_bleu = kwargs.pop('get_bleu', False)
        sampled_captions = self.model.build_sampler(max_len=15)    # (N, max_len, L), (N, max_len)
        
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            features_batch, image_files = sample_minibatch(data, self.batch_size)
            features_batch = data['features']
            image_files = data['file_names']
            # print(features_batch.shape)
            feed_dict = { self.model.feature: features_batch }
            sam_cap = sess.run([sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
            sam_cap = np.array(sam_cap).reshape((-1,15))
            decoded = decode_captions(sam_cap, self.model.idx_to_word)
            if get_bleu:
                save_pickle(decoded, "./data/test/test.predicted.captions.pkl")
                print('Bleu_1 score on test data: %s' % self.get_bleu(data_path='./data', split='test'))
            
            with open('output.txt', 'w') as f:
                for i, caption in enumerate(decoded):
                    if display:
                        print(caption)
                    f.write(image_files[i].split('/')[-1] + '#pred ' + caption[0].upper() + caption[1:] + '\n')


    def get_bleu(self, data_path, split):
        reference_path = os.path.join(data_path, "%s/%s.references.pkl" %(split, split))
        candidate_path = os.path.join(data_path, "%s/%s.predicted.captions.pkl" %(split, split))
        
        # load caption data
        with open(reference_path, 'rb') as f:
            ref = pickle.load(f)
        with open(candidate_path, 'rb') as f:
            cand = pickle.load(f)
        hypo = {}
        for i, caption in enumerate(cand):
            hypo[i] = [caption]
        
        # compute bleu score
        score, scores = Bleu(4).compute_score(ref, hypo)
        method = ["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]
        final_scores = {}
        for m, s in zip(method, score):
            final_scores[m] = s
        return final_scores['Bleu_1']


        

                