import tensorflow as tf
from utils.datafetcher_pix3d_initpc import *
from model.deform_baseline_1024 import Deform
import time
from progress.bar import IncrementalBar
from tqdm import trange
import pandas as pd
import sys
import os
from utils.visualize_utils import visualize_img_all_pix3d
#python test.py model_name

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
	batch_size = 1
	num_points = 1024
	num_concat = 1

	# Load data
	# data = DataFetcher('test')
	# data.setDaemon(True) ####
	# data.start()
	# train_number = data.iter

	#GPU settings 90% memory usage
	config = tf.ConfigProto() 
	config.gpu_options.per_process_gpu_memory_fraction = 0.9
	config.gpu_options.allow_growth = True

	all_cd = []
	all_emd = []
	all_cats = []
	csv_name = None

	with tf.Session(config = config) as sess:
		model = Deform(sess, 'evaluate', batch_size, num_concat=num_concat)
		ckpt = load_ckpt(model, model_dir)
		csv_name = '{}_{}k_pix3d.csv'.format(ckpt.model_checkpoint_path.split('.')[0], num_concat)
		print('Testing starts!')
		cats = ['table', 'sofa', 'chair']
		for cat in cats:
			print('Testing {}'.format(cat))
			models = get_pix3d_models(cat)
			batch = len(models)
			bar = IncrementalBar(max = batch)
			cat_cd = 0.0
			cat_emd = 0.0
			for i in trange(batch):
				# image, point = data.fetch()
				image, point, init_pc = fetch_batch_pix3d(models, i, batch_size, 256, num_concat)
				if point is None:
					continue
				# pred = model.predict(image)
				
				cd, emd, pred = model.evaluate(image, point, init_pc)
				visualize_img_all_pix3d(image, point, pred, i, batch_size, '{}_{}k_pix3d'.format(model_name, num_concat), cat)
				#scale metric
				cd *= 1000.
				emd /= 1000.
				print 'cd:', cd, 'emd', emd
				cat_cd += cd
				cat_emd += emd
				# bar.next()
			# bar.finish()
			cat_cd /= float(batch)
			cat_emd /= float(batch)
			all_cd.append(cat_cd)
			all_emd.append(cat_emd)
			all_cats.append(cat)
			print('{} cd: {}'.format(cat, cat_cd))
			print('{} emd: {}'.format(cat, cat_emd))
			# with open('{}.txt'.format(ckpt.model_checkpoint_path.split('.')[0]), 'a') as log:
			#     log.write('{} cd: {}\n'.format(shapenet_id_to_category[cat], cat_cd))
			#     log.write('{} emd: {}\n'.format(shapenet_id_to_category[cat], cat_emd))
	all_cats.append('mean')
	all_cd.append(sum(all_cd)/float((len(all_cd))))
	all_emd.append(sum(all_emd)/float((len(all_emd))))
	dataframe = pd.DataFrame({'cat':all_cats, 'cd':all_cd, 'emd':all_emd})
	dataframe.to_csv(csv_name, index = False, sep = ',')
	print('Testing finished!')      
