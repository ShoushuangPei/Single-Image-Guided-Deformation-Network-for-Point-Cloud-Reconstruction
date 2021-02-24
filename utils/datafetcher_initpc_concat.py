import os
import numpy as np
import h5py
import threading
import sys
if sys.version > '3':
	import queue as Queue
else:
	import Queue
import math

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

def init_pointcloud_loader(num_points):
    Z = np.random.rand(num_points) + 1.
    h = np.random.uniform(10., 246., size=(num_points,))
    w = np.random.uniform(10., 246., size=(num_points,))
    X = (w - 128) / 284. * -Z
    Y = (h - 128) / 284. * Z
    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1, 1))
    Z = np.reshape(Z, (-1, 1))
    XYZ = np.concatenate((X, Y, Z), 1)
    return XYZ.astype('float32')

def get_batch_init_pc(batch_size, num_points):
    batch = [init_pointcloud_loader(num_points) for i in range(batch_size)]
    return np.array(batch)

def get_batch_init_pc_concat(batch_size, num_points, num_concat):
    concat = [get_batch_init_pc(batch_size, num_points) for i in range(num_concat)]
    return concat

class DataFetcher(threading.Thread):
    def __init__(self, mode, batch_size = 32, epoch = 10, num_points = 256, num_concat=16):
        super(DataFetcher, self).__init__()
        self.stopped = False
        self.epoch = epoch
        self.current_epoch = 0
        self.queue = Queue.Queue(4)
        self.batch_size = batch_size
        self.num_points = num_points
        self.num_concat = num_concat
        self.mode = mode
        if self.mode == 'train':
            # self.image_path = '/media/tree/backup/projects/AttentionBased/data/train/image_192_256_12'
            self.image_path = '/media/tree/backup/projects/AttentionBased/data/train/image_256_256_12'
            #self.point_path = '/media/tree/backup/projects/AttentionBased/data/train/point_16384_12'
            self.point_path = '/media/tree/backup/projects/AttentionBased/data/train/point_1024_12'
        else:
            # self.image_path = '/media/tree/backup/projects/AttentionBased/data/test/image_192_256_12'
            self.image_path = '/media/tree/backup/projects/AttentionBased/data/test/image_256_256_12'
            #self.point_path = '/media/tree/backup/projects/AttentionBased/data/test/point_16384_12'
            self.point_path = '/media/tree/backup/projects/AttentionBased/data/test/point_1024_12'
        self.iter, self.cats_batches = self.calculate_cat_batch_number()
    
    def calculate_cat_batch_number(self):
        count = 0
        cats = shapenet_category_to_id.values()
        cat_batch_number = []
        for cat in cats:
            with h5py.File(os.path.join(self.image_path, '{}.h5'.format(cat)), 'r') as f:
                batch_number = f['image'].shape[0] / self.batch_size
                cat_batch_number.append(batch_number)
                count += batch_number
        cats_batches = dict(zip(cats, cat_batch_number))
        print(cats_batches)
        return count, cats_batches

    def run(self):
        if self.mode == 'train':
            while self.current_epoch < self.epoch:
                for cat, batch in self.cats_batches.iteritems():
                    with h5py.File(os.path.join(self.image_path, '{}.h5'.format(cat)), 'r') as fi:
                        with h5py.File(os.path.join(self.point_path, '{}.h5'.format(cat)), 'r') as fp:
                            for i in range(0, batch * self.batch_size, self.batch_size):
                                if self.stopped:
                                    break
                                self.queue.put((fi['image'][i:i+self.batch_size].astype('float32') / 255.0, fp['point'][i:i+self.batch_size], get_batch_init_pc_concat(self.batch_size, self.num_points, self.num_concat)))
                self.current_epoch += 1
        elif self.mode == 'predict':
            # for cat, batch in self.cats_batches.iteritems():
            #     with h5py.File(os.path.join(self.image_path, '{}.h5'.format(cat)), 'r') as fi:
            #         with h5py.File(os.path.join(self.point_path, '{}.h5'.format(cat)), 'r') as fp:
            #             for i in range(0, batch * self.batch_size, self.batch_size):
            #                 if self.stopped:
            #                     break 
            #                 self.queue.put((fi['point'][i:i+self.batch_size], fp['point'][i:i+self.batch_size]))

            cat = shapenet_category_to_id['chair']
            batch = self.cats_batches[cat]
            with h5py.File(os.path.join(self.image_path, '{}.h5'.format(cat)), 'r') as fi:
                with h5py.File(os.path.join(self.point_path, '{}.h5'.format(cat)), 'r') as fp:
                    for i in range(0, batch * self.batch_size, self.batch_size):
                        if self.stopped:
                            break 
                        self.queue.put((fi['image'][i:i+self.batch_size].astype('float32') / 255.0, fp['point'][i:i+self.batch_size], get_batch_init_pc_concat(self.batch_size, self.num_points, self.num_concat)))
        
        
        else:
            for cat, batch in self.cats_batches.iteritems():
                with h5py.File(os.path.join(self.image_path, '{}.h5'.format(cat)), 'r') as fi:
                    with h5py.File(os.path.join(self.point_path, '{}.h5'.format(cat)), 'r') as fp:
                        for i in range(0, batch * self.batch_size, self.batch_size):
                            if self.stopped:
                                break 
                            self.queue.put((fi['image'][i:i+self.batch_size].astype('float32') / 255.0, fp['point'][i:i+self.batch_size], get_batch_init_pc_concat(self.batch_size, self.num_points, self.num_concat)))

    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()
	
    def shutdown(self):
        self.stopped = True
        while not self.queue.empty():
            self.queue.get()



if __name__ == '__main__':
    data = DataFetcher('test',batch_size = 1)
    data.start()
    image, point = data.fetch()
    # current = 0

    # #create white background
    # background = np.zeros((256,256,3), dtype = np.uint8)
    # background.fill(255)

    # # 1. image (obj rendering) 
    # img = image[current, ...] * 255
    # img = img.astype('uint8')
    # img += background
    # img = np.where(img > 255 , img - 255, img)
    # cv2.imwrite('{:0>4}.png'.format(current), img)

    # # 3. gt_rendering
    # gt_rendering = background
    # X, Y, Z = point.T
    # F = 284
    # h = (-Y)/(-Z)*F + 256/2.0
    # w = X/(-Z)*F + 256/2.0
    # # h = np.minimum(np.maximum(h, 0), 255)
    # # w = np.minimum(np.maximum(w, 0), 255)
    # gt_rendering[np.round(h).astype(int), np.round(w).astype(int), 0] = 0
    # gt_rendering[np.round(h).astype(int), np.round(w).astype(int), 2] = 0
    # cv2.imwrite('{:0>4}.jpg'.format(current), gt_rendering)
    data.shutdown()
