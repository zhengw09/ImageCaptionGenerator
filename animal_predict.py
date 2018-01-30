# from resize import *
import tensorflow as tf
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.preprocessor import PreProcessor
from core.resize import resize_image
from core.utils import *
import os
import shutil
from PIL import Image

pred_folder = './image/pred'
resized_folder = './image/pred_resized'

# plz copy new images to pred_folder

def main():
    # check if there are new images in pred_folder
    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)
    new_images = []
    for img in os.listdir(pred_folder):
        if img[0] == '.':
            continue
        new_images.append(img)
    if new_images:
        shutil.rmtree(resized_folder)
        os.makedirs(resized_folder)
        for img in new_images:
            with open(os.path.join(pred_folder, img), 'r+b') as f:
                with Image.open(f) as image:
                    image = resize_image(image)
                    image.save(os.path.join(resized_folder, img))
        shutil.rmtree(pred_folder)
        os.makedirs(pred_folder)

    prep = PreProcessor(batch_size=100, max_length=15, word_count_threshold=1, cnn_model_path='data/imagenet-vgg-verydeep-19.mat')
    prep.run('pred')

    with open('./data/train/word_to_idx.pkl', 'rb') as f:
        word_to_idx = pickle.load(f)
    data = load_data(data_path='./data', split='pred')
    
    model = CaptionGenerator(word_to_idx, feature_dim=[196, 512], embed_dim=512, hidden_dim=1024, len_sent=16, lamda=1.0)   
    solver = CaptioningSolver(model, data, None, n_epochs=5, batch_size=128, learning_rate=0.001, print_every=1,\
     save_every=1, image_path='./image/pred_resized', model_path='./model/lstm', test_model='./model/lstm/model-5')

    solver.test(data)

    correct_cap = {'cat':0, 'dog':0}
    with open('output.txt') as f:
        for line in f.readlines():
            line = line.split('\n')[0]
            name = line.split('#')[0]
            cap = line.split('#')[1]
            animal = 'cat' if name[0] == 'C' else 'dog'
            if animal in cap:
                correct_cap[animal] += 1
    print('Accuracy on cat images: ' + str(correct_cap['cat']/float(100)))
    print('Accuracy on dog images: ' + str(correct_cap['dog']/float(100)))




if __name__ == '__main__':
    main()


















