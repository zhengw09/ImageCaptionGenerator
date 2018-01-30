from core.preprocessor import PreProcessor
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_data


def main():
    # preprocessing
    prep = PreProcessor(batch_size=100, max_length=15, word_count_threshold=1, cnn_model_path='data/imagenet-vgg-verydeep-19.mat')
    for split in ['train', 'dev']:
        prep.run(split, cat_img_num=0) # define the number of additional training cat images in the second argument

    # load train dataset
    train_data = load_data(data_path='./data', split='train')
    dev_data = load_data(data_path='./data', split='dev')
    word_to_idx = train_data['word_to_idx']

    model = CaptionGenerator(word_to_idx, feature_dim=[196, 512], embed_dim=512, hidden_dim=1024, len_sent=16, lamda=1.0)

    solver = CaptioningSolver(model, train_data, dev_data, n_epochs=5, batch_size=128, learning_rate=0.001, print_every=1000, save_every=5, \
     model_path='model/lstm/', test_model='model/lstm/model-5')

    solver.train()

if __name__ == "__main__":
    main()