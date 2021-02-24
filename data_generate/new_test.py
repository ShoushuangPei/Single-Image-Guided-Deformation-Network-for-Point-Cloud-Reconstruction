import numpy as np
import pickle
import os

output_folder = '/media/tree/backup/projects/AttentionBased/data/data'
data_prefix = '/media/tree/backup/graphXdata/ShapeNet/02691156/test'

data_path = []

data = os.listdir(data_prefix)
data_path = [os.path.join(data_prefix, i) for i in data]


def work(idx):
    # batch_img = [ np.load(self.image_path[idx+i]) for i in range(self.batch_size) ]
    # batch_label = [ np.load(self.point_path[idx+i]) for i in range(self.batch_size) ]
    # batch_model_id = []

    batch_img = []
    batch_label = []

    for i in range(24):

        data = data_path[idx+i]
        contents = pickle.load(open(data, 'rb'))
        # single_model_id = image_path.split('/')[-1]
        # image = cv2.imread(image_path)
        image_array = contents[0]
        point_array = contents[1]
        print(image_array.shape, point_array.shape)
        batch_img.append(image_array)
        batch_label.append(point_array)

        # batch_model_id.append(single_model_id)

    return np.array(batch_img), np.array(batch_label)
    # return np.array(batch_img), np.array(batch_label), batch_model_id

def fetch():
    idx = 0
    while idx <= len(data_path):
        work(idx)
        idx += 24

if __name__ == '__main__':
    import time
    load_start = time.time()
    idx = 0
    for i in range(3):
        image, point = work(idx)
        # print(image.shape, point.shape)
        idx += 24
    load_end = time.time()
    print('load_data', load_end - load_start)
