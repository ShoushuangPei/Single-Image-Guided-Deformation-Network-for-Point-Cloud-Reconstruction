import tensorflow as tf
from utils.datafetcher_initpc import DataFetcher
from model.deform import Deform
import time
import os
import sys
from tqdm import tqdm, trange
from utils.visualize_utils import visualize_train
import contextlib
#python train.py model_name

class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm"""
    file = None
    def __init__(self, file):
        self.file = file
    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

@contextlib.contextmanager
def stdout_redirect_to_tqdm():
    save_stdout = sys.stdout
    try:
        sys.stdout = DummyTqdmFile(sys.stdout)
        yield save_stdout
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout if necessary
    finally:
        sys.stdout = save_stdout

if __name__ == '__main__':
	model_name = sys.argv[1]
	model_dir = os.path.join('result', model_name)
	if not os.path.isdir(model_dir):
		os.makedirs(model_dir)
	train_log = os.path.join(model_dir,'{}_train.log'.format(model_name, ))
	
	#Global variables setting
	epoch = 50
	batch_size = 32

	# Load data
	data = DataFetcher('train', batch_size = batch_size, epoch = epoch)
	data.setDaemon(True)
	data.start()
	train_number = data.iter
	
	#GPU settings 90% memory usage
	config = tf.ConfigProto() 
	config.gpu_options.per_process_gpu_memory_fraction = 0.9
	config.gpu_options.allow_growth = True

	with tf.Session(config = config) as sess: 
		model = Deform(sess, 'train', batch_size)
		sess.run(tf.global_variables_initializer())
		start_epoch=0
		ckpt = tf.train.get_checkpoint_state(model_dir)
		if ckpt is not None:
			print ('loading '+ckpt.model_checkpoint_path + '  ....')
			model.saver.restore(sess, ckpt.model_checkpoint_path)
			start_epoch = int(ckpt.model_checkpoint_path.split('.')[0].split('_')[-1])

		print('Training starts!')
		for e in range(start_epoch, epoch):
			print('---- Epoch {}/{} ----'.format(e + 1, epoch))
			# model.saver.save(sess, os.path.join(model_dir, '{}_epoch_{}.ckpt'.format(model_name, e + 1)))
			with stdout_redirect_to_tqdm() as save_stdout:
				for i in tqdm(range(train_number), file=save_stdout, dynamic_ncols=True):
					image, point, init_pc = data.fetch()
					loss, cd, predicted_point = model.train_vis(image, point, init_pc)
					
					if i % 500 == 0:
						visualize_train(predicted_point, e, i, batch_size, model_name)

					if i % 100 == 0 or i == train_number - 1:
						current_time = time.strftime("%m-%d %H:%M:%S", time.localtime())
						print('Epoch {} / {} iter {} / {} --- Loss:{} - CD:{} - time:{}'.format(e + 1, epoch, i + 1, train_number, loss, cd, current_time))
						time.sleep(0.05)
						with open(train_log, 'a+') as f:
							f.write('Epoch {} / {} iter {} / {} --- Loss:{} - CD:{} - time:{}\n'.format(e + 1, epoch, i + 1, train_number, loss, cd, current_time))
			if (e + 1) % 5 == 0:
				model.saver.save(sess, os.path.join(model_dir, '{}_epoch_{}.ckpt'.format(model_name, e + 1)))
	data.shutdown()
	print('Training finished!')      

