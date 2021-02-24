import tensorflow as tf
from utils.datafetcher_initpc_concat import DataFetcher
from model.deform_baseline_1024 import Deform
import time
from tqdm import trange
import sys
import os
from utils.visualize_utils import visualize_img_all, visualize_img, visualize_feature_map, visualize_img_p2p

#python predict.py model_name

shapenet_id_to_category = {
'02691156': 'airplane',
'02828884': 'bench',
'02933112': 'cabinet',
'02958343': 'car',
'03001627': 'chair',
'03211117': 'monitor',
'03636649': 'lamp',
'03691459': 'speaker',
'04090263': 'rifle',
'04256520': 'sofa',
'04379243': 'table',
'04401088': 'telephone',
'04530566': 'vessel'
}

def load_ckpt(model, model_dir):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('No such file:[{}]'.format(model_dir))
    return ckpt

if __name__ == '__main__':
    model_name = sys.argv[1]
    model_dir = os.path.join('result', model_name)

    #Global variables setting
    batch_size = 24

    #Load data
    num_concat = 1
    data = DataFetcher('test', batch_size = batch_size, epoch = 1, num_points=256, num_concat=num_concat)
    data.setDaemon(True) ####
    data.start()
    train_number = data.iter

    #GPU settings 90% memory usage
    config = tf.ConfigProto() 
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        model = Deform(sess, 'predict', batch_size, num_concat=num_concat)
        ckpt = load_ckpt(model, model_dir)
        print('Predicting starts!')
        
        for cat, batch in data.cats_batches.iteritems():
            print('Predicting {}'.format(shapenet_id_to_category[cat]))
            start_time = time.strftime("%m-%d %H:%M:%S", time.localtime())
            print(start_time)
            for i in trange(batch):
                image, _point, init_pc = data.fetch()
                predicted_point = model.predict(image, init_pc)
                # visualize_img_all(image, _point, predicted_point, i, batch_size, model_name, shapenet_id_to_category[cat])
                visualize_img(predicted_point, i, batch_size, '{}_{}k'.format(model_name,num_concat), shapenet_id_to_category[cat])
                
    data.shutdown()
    print('Predicting finished!')      
