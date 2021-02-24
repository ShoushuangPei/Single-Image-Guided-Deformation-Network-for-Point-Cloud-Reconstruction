import tensorflow as tf
from utils.datafetcher_initpc_concat import DataFetcher
from model.deform import Deform
import time
import os
import sys
from progress.bar import IncrementalBar
from tqdm import trange
import pandas as pd


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
	
	batch_size = 4

	# Load data
	num_concat = 16
	data = DataFetcher('test', batch_size = batch_size,num_points=1024, num_concat=num_concat)
	data.setDaemon(True)
	data.start()
	
	#GPU settings 90% memory usage
	config = tf.ConfigProto() 
	config.gpu_options.per_process_gpu_memory_fraction = 0.9
	config.gpu_options.allow_growth = True

	all_cd = []
	all_emd = []
	all_cats = []
	csv_name = None

	with tf.Session(config = config) as sess:    
		model = Deform(sess, 'test_concat', batch_size, num_concat=num_concat)
		ckpt = load_ckpt(model, model_dir)
		csv_name = '{}_{}k.csv'.format(ckpt.model_checkpoint_path.split('.')[0], num_concat)
		print('Testing starts!')
		for cat, batch in data.cats_batches.iteritems():
			print('Testing {}'.format(shapenet_id_to_category[cat]))
			# bar = IncrementalBar(max = batch)
			cat_cd = 0.0
			cat_emd = 0.0
			for i in trange(batch):
				image, point, init_pc = data.fetch()
				cd, emd = model.test_concat(image, point, init_pc)
				# cd = model.test(image, point)
				#scale metric
				cd *= 1000.
				emd /= 1000.
				cat_cd += cd
				cat_emd += emd
				# bar.next()
			# bar.finish()
			cat_cd /= float(batch)
			cat_emd /= float(batch)
			all_cd.append(cat_cd)
			all_emd.append(cat_emd)
			all_cats.append(shapenet_id_to_category[cat])
			print('{} cd: {}'.format(shapenet_id_to_category[cat], cat_cd))
			print('{} emd: {}'.format(shapenet_id_to_category[cat], cat_emd))
			
	data.shutdown()
	all_cats.append('mean')
	all_cd.append(sum(all_cd)/float((len(all_cd))))
	all_emd.append(sum(all_emd)/float((len(all_emd))))
	dataframe = pd.DataFrame({'cat':all_cats, 'cd':all_cd, 'emd':all_emd})
	dataframe.to_csv(csv_name, index = False, sep = ',')
	print('Testing finished!')         

