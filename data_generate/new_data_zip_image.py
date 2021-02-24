import numpy as np
import os
import cv2
from progress.bar import IncrementalBar
import time
import h5py
import json
import re
import sys
from tqdm import trange

shapenet_category_to_id = {
'airplane'	: '02691156',
'bench'		: '02828884',
'cabinet'	: '02933112',
'car'		: '02958343',
'chair'		: '03001627',
'lamp'		: '03636649',
'monitor'	: '03211117',
'rifle'		: '04090263',
'sofa'		: '04256520',
'speaker'	: '03691459',
'table'		: '04379243',
'telephone'	: '04401088',
'vessel'	: '04530566'
}


def dump_image_point():
    image_prefix = '/media/tree/data1/projects/PointGAN/3d-lmnet/data'
    point_prefix = '/media/tree/data1/projects/AttentionBased/data'
    train_output_folder = '/media/tree/backup/projects/new_work/data/train'
    test_output_folder = '/media/tree/backup/projects/new_work/data/test'
    image_input_folder = 'ShapeNetRendering'
    image_output_folder = 'image256'
    image_number = 24

    with open('/media/tree/backup/projects/new_work/data/train_models.json', 'r') as f:
        train_models_dict = json.load(f)

    with open('/media/tree/backup/projects/new_work/data/test_models.json', 'r') as f:
        test_models_dict = json.load(f)

    cats = shapenet_category_to_id.values()
    for cat in cats:
        print(cat, 'starts at ', time.strftime("%m-%d %H:%M:%S", time.localtime()))
        print(cat, 'loading train_split!')
        train_image_models = []
        train_img_path = []
        train_image_models.extend([os.path.join(image_prefix, image_input_folder, model, 'rendering') for model in train_models_dict[cat]])
        for each in train_image_models:
            for index in range(image_number):
                train_img_path.append(os.path.join(each, '{0:02d}.png'.format(int(index))))
                
        print(cat, 'train_split loaded!')

        train_image_save = h5py.File(os.path.join(train_output_folder, image_output_folder, '{}.h5'.format(cat)), mode = 'w')
        
        train_img_shape = (len(train_img_path), 256, 256, 3)

        train_image_save.create_dataset('image', train_img_shape, np.uint8)
        
        print(cat, 'saving train data at', time.strftime("%m-%d %H:%M:%S", time.localtime()))
        # train_bar =  IncrementalBar(max=len(train_img_path))
        for i in trange(len(train_img_path)):
            image_array= load_data(train_img_path[i])
            train_image_save['image'][i, ...] = image_array
            # train_bar.next()
        # train_bar.finish()
        print(cat, 'train data saved!')
        
        train_image_save.close()

        print(cat, 'loading test_split!')
        test_image_models = []
        test_img_path = []
        test_image_models.extend([os.path.join(image_prefix, image_input_folder, model, 'rendering') for model in test_models_dict[cat]])
        for each in test_image_models:
            for index in range(image_number):
                test_img_path.append(os.path.join(each, '{0:02d}.png'.format(int(index))))

        print(cat, 'test_split loaded!')

        test_image_save = h5py.File(os.path.join(test_output_folder, image_output_folder, '{}.h5'.format(cat)), mode = 'w')
        
        test_img_shape = (len(test_img_path), 256, 256, 3)
        
        test_image_save.create_dataset('image', test_img_shape, np.uint8)

        print(cat, 'saving test data at ', time.strftime("%m-%d %H:%M:%S", time.localtime()))
        # test_bar =  IncrementalBar(max=len(test_img_path))
        for i in trange(len(test_img_path)):
            image_array = load_data(test_img_path[i])
            test_image_save['image'][i, ...] = image_array
            # test_bar.next()
        # test_bar.finish()
        print(cat, 'test data saved!')
        
        print(cat, 'finished at ', time.strftime("%m-%d %H:%M:%S", time.localtime()))
        
        test_image_save.close()


def load_data(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image[np.where(image[:, :, 3] == 0)] = 255
    image = cv2.resize(image, (256, 256))
    image_array = np.array(image[:, :, :3])
    image_array = image_array.astype(np.uint8)

    return image_array

# def load_single():
#     data_prefix = '/media/tree/data1/projects/AttentionBased/data'
#     with open('/media/tree/data1/projects/PointGAN/3d-lmnet/data/splits/train_models.json', 'r') as f:
#         train_models_dict = json.load(f)
#     train_image_models = []
#     train_img_path = []
#     train_image_models.extend([os.path.join(data_prefix, 'image_png', model) for model in train_models_dict['04090263']])
#     for each in train_image_models:
#         for index in range(24):
#             train_img_path.append(os.path.join(each, '{0:02d}.png'.format(int(index))))
#     for i in range(len(train_img_path)):
#         image = cv2.imread(train_img_path[i])
#         print(train_img_path[i])
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    dump_image_point()
