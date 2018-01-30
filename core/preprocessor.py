from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg
from core.utils import *
from shutil import copyfile
from random import shuffle

import tensorflow as tf
import numpy as np
import pandas as pd
import hickle
import os
import json


class PreProcessor(object):

    def __init__(self, batch_size, max_length=15, word_count_threshold=1, cnn_model_path='data/imagenet-vgg-verydeep-19.mat'):
        self.batch_size = batch_size
        self.max_length = max_length
        self.word_count_threshold = word_count_threshold
        self.cnn_model_path = cnn_model_path


    def _get_json(self, split, cat_img_num=0):
        j = {}
        j['annotations'] = []
        if split != 'pred':
            img_to_cap = {}
            with open('data/Flickr8k.token.txt') as f:
                for line in f.readlines():
                    line = line.split('\n')[0]
                    img_name = line.split('#')[0]
                    caption = line.split('#')[1][2:]
                    if img_name not in img_to_cap:
                        img_to_cap[img_name] = []
                    img_to_cap[img_name].append(caption)
            
            # check if cat images needed
            cat_img_names = []
            if cat_img_num > 0:
                cat_list = os.listdir('image/cat_resized')
                for name in cat_list:
                    if name[0] == '.':
                        continue
                    cat_img_names.append(name)
                    src = os.path.join('./image/cat_resized', name)
                    dst = os.path.join('./image/train_resized', name)
                    copyfile(src, dst)
                    if len(cat_img_names) == cat_img_num:
                        break


                with open('data/cat_annotations.txt') as f:
                    for line in f.readlines():
                        line = line.split('\n')[0]
                        img_name = line.split('#')[0]
                        caption = line.split('#')[1][2:]
                        if img_name in cat_img_names:
                            if img_name not in img_to_cap:
                                img_to_cap[img_name] = []
                            img_to_cap[img_name].append(caption)

            images_file = 'data/Flickr_8k.%s.txt' % split
            image_filenames = set()
            with open(images_file) as f:
                for line in f.readlines():
                    name = line.split('\n')[0]
                    image_filenames.add(name)
            for name in cat_img_names:
                image_filenames.add(name)

            image_filenames = list(image_filenames)
            shuffle(image_filenames)
            for img in image_filenames:
                if img[0] == '.':
                    continue
                for cap in img_to_cap[img]:
                    ann = {}
                    ann['file_name'] = img
                    ann['caption'] = cap
                    j['annotations'].append(ann)

        else:
            image_filenames = os.listdir('image/pred_resized')
            for img in image_filenames:
                if img[0] == '.':
                    continue
                ann = {}
                ann['file_name'] = img
                ann['caption'] = ''
                j['annotations'].append(ann)

        with open('data/annotations/captions_%s.json' % split, 'w') as outfile:
            json.dump(j, outfile)


    def _process_caption_data(self, caption_file, image_dir):
        # return a panda datafram, with colomn 'file_name' and 'caption'
        with open(caption_file) as f:
            caption_data = json.load(f)

        # data is a list of dictionary which contains 'captions' and 'file_name' as keys.
        data = []
        for annotation in caption_data['annotations']:
            annotation['file_name'] = os.path.join(image_dir, annotation['file_name'])
            data.append(annotation)
        
        # convert to pandas dataframe (for later visualization or debugging)
        caption_data = pd.DataFrame.from_dict(data)
        # caption_data.sort_values(by='file_name', inplace=True)
        caption_data = caption_data.reset_index(drop=True)
        
        del_idx = []
        for i, caption in enumerate(caption_data['caption']):
            caption = caption.replace('.','').replace(',','').replace("'","").replace('"','')
            caption = caption.replace('&','and').replace('(','').replace(")","").replace('-',' ')
            caption = " ".join(caption.split())  # replace multiple spaces
            
            caption_data.at[i, 'caption'] = caption.lower()
            if len(caption.split(" ")) > self.max_length:
                del_idx.append(i)
        
        # delete captions if size is larger than max_length
        print "The number of captions before deletion: %d" %len(caption_data)
        caption_data = caption_data.drop(caption_data.index[del_idx])
        caption_data = caption_data.reset_index(drop=True)
        print "The number of captions after deletion: %d" %len(caption_data)
        return caption_data


    def _build_vocab(self, train_dataset):
        # return a dict, mapping word to idx
        counter = Counter()
        max_len = 0
        for i, caption in enumerate(train_dataset['caption']):
            words = caption.split(' ') # caption contrains only lower-case words
            for w in words:
                counter[w] +=1
            
            if len(caption.split(" ")) > max_len:
                max_len = len(caption.split(" "))

        vocab = [word for word in counter if counter[word] >= self.word_count_threshold]
        print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), self.word_count_threshold))

        word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
        idx = 3
        for word in vocab:
            word_to_idx[word] = idx
            idx += 1
        print "Max length of caption: ", max_len
        return word_to_idx


    def _build_caption_vector(self, annotations, word_to_idx):
        # return a 2-D array, each row be a caption vector
        n_examples = len(annotations)
        captions = np.ndarray((n_examples, self.max_length+2)).astype(np.int32)   

        for i, caption in enumerate(annotations['caption']):
            words = caption.split(" ") # caption contrains only lower-case words
            cap_vec = []
            cap_vec.append(word_to_idx['<START>'])
            for word in words:
                if word in word_to_idx:
                    cap_vec.append(word_to_idx[word])
            cap_vec.append(word_to_idx['<END>'])
            
            # pad short caption with the special null token '<NULL>' to make it fixed-size vector
            if len(cap_vec) < (self.max_length + 2):
                for j in range(self.max_length + 2 - len(cap_vec)):
                    cap_vec.append(word_to_idx['<NULL>']) 
            
            captions[i, :] = np.asarray(cap_vec)
        print "Finished building caption vectors"
        return captions


    def _build_file_names(self, annotations):
        # return a 1-D array of unique file names
        file_names = np.asarray(annotations['file_name'].unique())
        return file_names


    def _build_image_idxs(self, annotations):
        # return a 1-D array of images idxs, e.g. [0,0,0,0,1,1,1,1,1,2...]
        # and a dic: {idx: [cap1, cap2, ...]}
        image_idxs = np.ndarray(len(annotations), dtype=np.int32)
        image_cap = {}
        name_to_idx = {}
        captions = annotations['caption']
        for i, image_name in enumerate(annotations['file_name']):
            if image_name not in name_to_idx:
                name_to_idx[image_name] = len(name_to_idx)
                image_cap[name_to_idx[image_name]] = []
            image_idxs[i] = name_to_idx[image_name]
            image_cap[name_to_idx[image_name]].append(captions[i])
        return image_idxs, image_cap


    def run(self, split, cat_img_num=0):
        self._get_json(split, cat_img_num)
        caption_file='data/annotations/captions_%s.json' % split
        image_dir='image/%s_resized/' % split
        annotations = self._process_caption_data(caption_file, image_dir)
        save_pickle(annotations, 'data/%s/%s.annotations.pkl' % (split, split)) 
        # build vocabulary
        if split == 'train':
            word_to_idx = self._build_vocab(annotations)
            save_pickle(word_to_idx, 'data/train/word_to_idx.pkl')
        else:
            word_to_idx = load_pickle('data/train/word_to_idx.pkl')

        cap_vectors = self._build_caption_vector(annotations=annotations, word_to_idx=word_to_idx)
        save_pickle(cap_vectors, './data/%s/%s.captions.pkl' % (split, split))

        file_names = self._build_file_names(annotations)
        save_pickle(file_names, './data/%s/%s.file.names.pkl' % (split, split))

        image_idxs, image_cap = self._build_image_idxs(annotations)
        save_pickle(image_idxs, './data/%s/%s.image.idxs.pkl' % (split, split))
        save_pickle(image_cap, './data/%s/%s.references.pkl' % (split, split))
        print "Finished building %s caption dataset" %split

        # extract conv5_3 feature vectors
        vggnet = Vgg(self.cnn_model_path)
        vggnet.build()
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            save_path = './data/%s/%s.features.hkl' % (split, split)
            image_path = list(annotations['file_name'].unique())
            n_examples = len(image_path)
            print('Total image number: %s' % (n_examples))

            all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)

            for start in range(0, n_examples, self.batch_size):
                end = start + self.batch_size
                image_batch_file = image_path[start:end]
                image_batch = np.array(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file)).astype(
                    np.float32)
                feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
                all_feats[start:end, :] = feats
                print ("Processed %d %s features.." % (end, split))

            # use hickle to save huge feature vectors
            hickle.dump(all_feats, save_path)
            print ("Saved %s.." % (save_path))
        tf.reset_default_graph()


