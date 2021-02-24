from os.path import join, exists, isdir, dirname, abspath, basename
import cv2
import numpy as np
import json
import re
import ctypes as ct
import sklearn.preprocessing

dll = np.ctypeslib.load_library('render_balls_so','./utils')
HEIGHT = 256
WIDTH = 256
PAD = 80

data_dir_imgs = '/media/tree/data1/data/pix3d'
# data_dir_pcl = '/media/tree/data1/data/pix3d'
data_dir_pcl = '/media/tree/data1/data/data/pix3d/pointclouds'

view_path = '/media/tree/data1/projects/AttentionBased/PSGN/data_generate/pix3d_view.txt'
cam_params = np.loadtxt(view_path)
test_png = '/media/tree/data1/projects/3d-lmnet/test.png'

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

def get_pix3d_models(cat):

	with open(join(data_dir_imgs, 'pix3d.json'), 'r') as f:
		models_dict = json.load(f)
	models = []

	# cats = ['chair','sofa','table']
	cats = []
	cats.append(cat)
	
	# Check for truncation and occlusion before adding a model to the evaluation list
	for d in models_dict:
		if d['category'] in cats:
			if not d['truncated'] and not d['occluded'] and not d['slightly_occluded']:
				models.append(d)

	print 'Total models = {}\n'.format(len(models))
	return models
		
def camera_info_pix3d(coord):
	# theta = np.deg2rad(param[0])
	# phi = np.deg2rad(param[1])

	# camY = param[3]*np.sin(phi)
	# temp = param[3]*np.cos(phi)
	# camX = temp * np.cos(theta)    
	# camZ = temp * np.sin(theta)
	camX = coord[0]
	camY = coord[1]
	camZ = coord[2]
	cam_pos = np.array([camX, camY, camZ])        

	axisZ = cam_pos.copy()
	axisY = np.array([0,1,0])
	axisX = np.cross(axisY, axisZ)
	axisY = np.cross(axisZ, axisX)

	cam_mat = np.array([axisX, axisY, axisZ])
	cam_mat = sklearn.preprocessing.normalize(cam_mat, axis=1)
	return cam_mat, cam_pos		


def fetch_batch_pix3d(models, batch_num, batch_size, num_points, num_concat):
	''' 
	Inputs:
		models: List of pix3d dicts
		batch_num: batch_num during epoch
		batch_size: batch size for training or validation
	Returns:
		batch_ip: input RGB image of shape (B, HEIGHT, WIDTH, 3)
		batch_gt: gt point cloud of shape (B, NUM_POINTS, 3)
	Description:
		Batch Loader for Pix3D dataset
	'''
	batch_ip = []
	batch_gt = []

	for ind in xrange(batch_num*batch_size,batch_num*batch_size+batch_size):
		_dict = models[ind]
		model_path = '/'.join(_dict['model'].split('/')[:-1])
		model_name = re.search('model(.*).obj', _dict['model'].strip().split('/')[-1]).group(1)
		img_path = join(data_dir_imgs, _dict['img'])
		mask_path = join(data_dir_imgs, _dict['mask'])
		bbox = _dict['bbox'] # [width_from, height_from, width_to, height_to]
		pcl_path_1K = join(data_dir_imgs, _dict['model'].split('.')[0] + '.npy')
		# pcl_path_1K = join(data_dir_pcl, model_path,'pcl_1024.npy')
		ip_image = cv2.imread(img_path)
		# ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)
		ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2GRAY)
		ip_image = cv2.cvtColor(ip_image, cv2.COLOR_GRAY2RGB)
		mask_image = cv2.imread(mask_path)!=0
		ip_image=ip_image*mask_image
		ip_image = ip_image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

		current_size = ip_image.shape[:2] # current_size is in (height, width) format
		ratio = float(HEIGHT-PAD)/max(current_size)
		new_size = tuple([int(x*ratio) for x in current_size])
		ip_image = cv2.resize(ip_image, (new_size[1], new_size[0])) # new_size should be in (width, height) format
		delta_w = WIDTH - new_size[1]
		delta_h = HEIGHT - new_size[0]
		top, bottom = delta_h//2, delta_h-(delta_h//2)
		left, right = delta_w//2, delta_w-(delta_w//2)
		color = [0, 0, 0]
		ip_image = cv2.copyMakeBorder(ip_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
		# background = np.zeros((256,256,3), dtype = np.uint8)
		# background.fill(255)
		# ip_image += background
		# ip_image = np.where(ip_image > 255 , ip_image - 255, ip_image)
		# cv2.imwrite(test_png, ip_image)
		ip_image = ip_image.astype('float32') / 255.0
		
		if exists(pcl_path_1K):
			position = np.load(pcl_path_1K)
			# pcl_gt = rotate(rotate(np.load(pcl_path_1K), xangle, yangle), xangle)
			# for index, param in enumerate(cam_params):
			# 	camera tranform
			cam_pix3d = _dict['cam_position']
			cam_mat, cam_pos = camera_info_pix3d(cam_pix3d)
			# print cam_pos

				# pt_trans_16k = np.dot(position_16k-cam_pos, cam_mat.transpose())
			pt_trans = np.dot(position-cam_pos, cam_mat.transpose())
				# nom_trans = np.dot(normal, cam_mat.transpose())
				# train_data = np.hstack((pt_trans, nom_trans))
				# train_data = pt_trans
				
				# img_path = os.path.join(os.path.split(view_path)[0], '%02d.png'%index)
				# np.savetxt(img_path.replace('png','xyz'), train_data)

				# np.savetxt(img_path.replace('png','xyz'), pt_trans_16k)
				# np.savetxt(img_path.replace('png','xyz'), pt_trans)


			# pcl_gt = rotate(rotate(np.load(pcl_path_1K), xangle, yangle), xangle)
			
			
			
			# pcl_gt = np.load(pcl_path_1K)
			pcl_gt = pt_trans
			batch_gt.append(pcl_gt)
			batch_ip.append(ip_image)
			batch_ip = np.array(batch_ip)
			batch_gt = np.array(batch_gt)

		else:
			batch_gt = None
			batch_ip = None
			print pcl_path_1K


	return batch_ip, batch_gt, get_batch_init_pc_concat(batch_size, num_points, num_concat)

def get_rendering(point, ballradius=2, background=(255,255,255), image_size=256):
	point=point-point.mean(axis=0)
	radius=((point**2).sum(axis=-1)**0.5).max()
	point/=(radius*2.2)/image_size

	c0=np.zeros((len(point),),dtype='float32')+242 #G
	c1=np.zeros((len(point),),dtype='float32')+248 #R
	c2=np.zeros((len(point),),dtype='float32')+220 #B

	c0=np.require(c0,'float32','C')
	c1=np.require(c1,'float32','C')
	c2=np.require(c2,'float32','C')

	show=np.zeros((image_size,image_size,3),dtype='uint8')
	def render():
		npoint = point + [image_size/2,image_size/2,0]
		# npoint = point
		ipoint = npoint.astype('int32')
		show[:]=background
		dll.render_ball(
			ct.c_int(show.shape[0]),
			ct.c_int(show.shape[1]),
			show.ctypes.data_as(ct.c_void_p),
			ct.c_int(ipoint.shape[0]),
			ipoint.ctypes.data_as(ct.c_void_p),
			c0.ctypes.data_as(ct.c_void_p),
			c1.ctypes.data_as(ct.c_void_p),
			c2.ctypes.data_as(ct.c_void_p),
			ct.c_int(ballradius)
		)

	render()
	return show

if __name__ == '__main__':
	chair = get_pix3d_models('table')
	iters = len(chair)
	for cnt in range(1):
		ip, gt = fetch_batch_pix3d(chair, cnt, 1)
		# ip = np.squeeze(ip,0)
		# cv2.imwrite('/media/tree/data1/projects/3d-lmnet/test.png', ip)
		gt = np.squeeze(gt,0)
		np.savetxt('/media/tree/data1/projects/3d-lmnet/test.xyz', gt)
		gt = np.expand_dims(gt,0)
		X, Y, Z = gt.T
		pred_point_t = np.concatenate([-Y,X,Z],1)
		pred_rendering = get_rendering(np.vstack(pred_point_t))
		cv2.imwrite('/media/tree/data1/projects/3d-lmnet/test.png', pred_rendering)
		print ip.shape, gt.shape
