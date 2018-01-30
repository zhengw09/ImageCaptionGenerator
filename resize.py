from PIL import Image
import os
from core.resize import resize_image

def main():
    splits = ['train', 'val', 'test']
    folder = './image/Flicker8k_images'
    for split in splits:
        resized_folder = './image/%s_resized/' % split
        if not os.path.exists(resized_folder):
            os.makedirs(resized_folder)
        print 'Start resizing %s images.' % split

        image_files = []
        listfolder = 'data/Flickr_8k.%s.txt' % split
        with open(listfolder) as g:
            for line in g.readlines():
                image_files.append(line.split('\n')[0])
        
        num_images = len(image_files)
        for i, image_file in enumerate(image_files):
            with open(os.path.join(folder, image_file), 'r+b') as f:
                with Image.open(f) as image:
                    image = resize_image(image)
                    image.save(os.path.join(resized_folder, image_file), image.format)
            if i % 100 == 0:
                print 'Resized images: %d/%d' %(i, num_images)
              
            
if __name__ == '__main__':
    main()