# from resize import *
import tensorflow as tf
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.preprocessor import PreProcessor
from core.resize import resize_image
from core.utils import *
import os
import shutil
import json

def main():

    prep = PreProcessor(batch_size=100, max_length=15, word_count_threshold=1, cnn_model_path='data/imagenet-vgg-verydeep-19.mat')
    prep.run('test')

    with open('./data/train/word_to_idx.pkl', 'rb') as f:
        word_to_idx = pickle.load(f)
    data = load_data(data_path='./data', split='test')
    
    model = CaptionGenerator(word_to_idx, feature_dim=[196, 512], embed_dim=512, hidden_dim=1024, len_sent=16, lamda=1.0)   
    solver = CaptioningSolver(model, data, None, n_epochs=20, batch_size=128, learning_rate=0.001, print_every=1,\
     save_every=1, image_path='./image/test_resized', model_path='./model/lstm', test_model='./model/lstm/model-5')

    solver.test(data, get_bleu=True)

    caption_file='data/annotations/captions_test.json'
    with open(caption_file) as f:
        groundtruth = json.load(f)
    with open('groundtruth.txt', 'w') as f:
        for i, ann in enumerate(groundtruth['annotations']):
            f.write(ann['file_name'] + '  ' + ann['caption'] + '\n')
            if (i+1)%5 == 0:
                f.write('\n')




if __name__ == '__main__':
    main()


















