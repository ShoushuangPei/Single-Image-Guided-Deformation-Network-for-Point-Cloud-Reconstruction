import tensorflow as tf
import tflearn
from layer.basic_layers import ResidualBlock
from layer.attention_module import AttentionModule
from metric.tf_nndistance import nn_distance
from metric.tf_approxmatch import approx_match, match_cost
import numpy as np

def scale(gt, pred): #pr->[-0.5,0.5], gt->[-0.5,0.5]
    '''
    Scale the input point clouds between [-max_length/2, max_length/2]
    '''

    # Calculate min and max along each axis x,y,z for all point clouds in the batch
    min_gt = tf.convert_to_tensor([tf.reduce_min(gt[:,:,i], axis=1) for i in xrange(3)]) #(3, B)
    max_gt = tf.convert_to_tensor([tf.reduce_max(gt[:,:,i], axis=1) for i in xrange(3)]) #(3, B)
    min_pr = tf.convert_to_tensor([tf.reduce_min(pred[:,:,i], axis=1) for i in xrange(3)]) #(3, B)
    max_pr = tf.convert_to_tensor([tf.reduce_max(pred[:,:,i], axis=1) for i in xrange(3)]) #(3, B)

    # Calculate x,y,z dimensions of bounding cuboid
    length_gt = tf.abs(max_gt - min_gt) #(3, B)
    length_pr = tf.abs(max_pr - min_pr) #(3, B)

    # Calculate the side length of bounding cube (maximum dimension of bounding cuboid)
    # Then calculate the delta between each dimension of the bounding cuboid and the side length of bounding cube
    diff_gt = tf.reduce_max(length_gt, axis=0, keep_dims=True) - length_gt #(3, B)
    diff_pr = tf.reduce_max(length_pr, axis=0, keep_dims=True) - length_pr #(3, B)

    # Pad the xmin, xmax, ymin, ymax, zmin, zmax of the bounding cuboid to match the cuboid side length
    new_min_gt = tf.convert_to_tensor([min_gt[i,:] - diff_gt[i,:]/2. for i in xrange(3)]) #(3, B)
    new_max_gt = tf.convert_to_tensor([max_gt[i,:] + diff_gt[i,:]/2. for i in xrange(3)]) #(3, B)
    new_min_pr = tf.convert_to_tensor([min_pr[i,:] - diff_pr[i,:]/2. for i in xrange(3)]) #(3, B)
    new_max_pr = tf.convert_to_tensor([max_pr[i,:] + diff_pr[i,:]/2. for i in xrange(3)]) #(3, B)

    # Compute the side length of bounding cube
    size_pr = tf.reduce_max(length_pr, axis=0) #(B,)
    size_gt = tf.reduce_max(length_gt, axis=0) #(B,)

    # Calculate scaling factor according to scaled cube side length (here = 2.)
    scaling_factor_gt = 1. / size_gt #(B,)
    scaling_factor_pr = 1. / size_pr #(B,)

    # Calculate the min x,y,z coordinates for the scaled cube (here = (-1., -1., -1.))
    box_min = tf.ones_like(new_min_gt) * -0.5 #(3, B)

    # Calculate the translation adjustment factor to match the minimum coodinates of the scaled cubes
    adjustment_factor_gt = box_min - scaling_factor_gt * new_min_gt #(3, B)
    adjustment_factor_pr = box_min - scaling_factor_pr * new_min_pr #(3, B)

    # Perform scaling then translation to the point cloud ? verify this
    pred_scaled = tf.transpose((tf.transpose(pred) * scaling_factor_pr)) + tf.reshape(tf.transpose(adjustment_factor_pr), (-1,1,3))
    gt_scaled   = tf.transpose((tf.transpose(gt) * scaling_factor_gt)) + tf.reshape(tf.transpose(adjustment_factor_gt), (-1,1,3))

    return gt_scaled, pred_scaled

class Deform(object):
    def __init__(self, sess, mode, batch_size):
        self.sess = sess
        self.mode = mode
        self.batch_size = batch_size

        self.attention_module = AttentionModule()
        self.residual_block = ResidualBlock()
        #initialize input and output
        self.image = tf.placeholder(tf.float32, shape = (self.batch_size, 256, 256, 3), name = 'image')
        self.point = tf.placeholder(tf.float32, shape = (self.batch_size, 16384, 3), name = 'point')
        self.init_pc = tf.placeholder(tf.float32, shape = (self.batch_size, 250, 3), name = 'init_pc')
        # self.point = tf.placeholder(tf.float32, shape = (None, 1024, 3), name = 'point')
        self.gt = None
        self.pred = None

        #initialize before build graph
        self.train_loss = 0
        self.cd_loss = 0
        self.emd_loss = 0
        self.optimizer = None
        self.opt_op = None

        self.build()
        #initialize saver after build graph
        self.saver = tf.train.Saver(max_to_keep=50)


    def build(self):
        if self.mode == 'train':
            self.build_graph()
            self.build_optimizer()
        elif self.mode == 'test':
            self.build_graph(is_training=False)
            self.build_loss_calculater()
        elif self.mode == 'evaluate':
            self.build_graph(is_training=False)
            self.build_loss()
        else:
            self.build_graph(is_training=False)

    def build_optimizer(self):
        print('Building chamfer distance loss optimizer...')

        self.optimizer = tf.train.AdamOptimizer(3e-5)
        dist1,idx1,dist2,idx2 = nn_distance(self.gt, self.pred)
        loss_nodecay = (tf.reduce_mean(dist1) + 0.55 * tf.reduce_mean(dist2)) * 10000
        self.train_loss = loss_nodecay + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) * 0.1
        self.cd_loss = tf.reduce_mean(dist1) + tf.reduce_mean(dist2)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.opt_op = self.optimizer.minimize(self.train_loss)

    def build_loss_calculater(self):
        if self.mode == 'test':
            self.gt, self.pred = scale(self.gt, self.pred)
        #cd
        dist1,idx1,dist2,idx2 = nn_distance(self.gt, self.pred)
        self.cd_loss = tf.reduce_mean(dist1) + tf.reduce_mean(dist2)
        #emd
        match = approx_match(self.pred, self.gt)
        # self.emd_loss = tf.reduce_mean(match_cost(self.gt, self.pred, match)) / float(tf.shape(self.pred)[1])
        self.emd_loss = tf.reduce_mean(match_cost(self.pred, self.gt, match))
    
    def build_loss(self):
        
        self.scaled_gt, self.scaled_pred = scale(self.gt, self.pred)

        #cd
        dist1,idx1,dist2,idx2 = nn_distance(self.scaled_gt, self.scaled_pred)
        self.cd_loss = tf.reduce_mean(dist1) + tf.reduce_mean(dist2)
        #emd
        match = approx_match(self.scaled_pred, self.scaled_gt)
        # self.emd_loss = tf.reduce_mean(match_cost(self.gt, self.pred, match)) / float(tf.shape(self.pred)[1])
        self.emd_loss = tf.reduce_mean(match_cost(self.scaled_pred, self.scaled_gt, match))


    def train(self, image, point, init_pc):
        _, train_loss, cd_loss = self.sess.run([self.opt_op, self.train_loss, self.cd_loss], feed_dict = {self.image : image, self.point : point, self.init_pc : init_pc})
        return train_loss, cd_loss
    
    def train_vis(self, image, point, init_pc):
        _, train_loss, cd_loss, pred = self.sess.run([self.opt_op, self.train_loss, self.cd_loss, self.pred], feed_dict = {self.image : image, self.point : point, self.init_pc : init_pc})
        return train_loss, cd_loss, pred
    
    def test(self, image, point, init_pc):
        cd, emd = self.sess.run([self.cd_loss, self.emd_loss], feed_dict = {self.image : image, self.point : point, self.init_pc : init_pc})
        return cd, emd
    
    def predict(self, image, init_pc):
        predicted_pointcloud = self.sess.run(self.pred, feed_dict = {self.image : image, self.init_pc : init_pc})
        return predicted_pointcloud

    def evaluate(self, image, point, init_pc):		
        cd, emd, pred = self.sess.run([self.cd_loss, self.emd_loss, self.scaled_pred], feed_dict={self.image : image, self.point : point, self.init_pc : init_pc})
        return cd, emd, pred


    def build_graph(self, is_training = True):
        is_training = tf.cast(is_training, tf.bool)
        # image encoder
        x=self.image
        #256 256
        x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
        #128 128
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
        # 64,64,64
        x1=x
        #64 64
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
        # 128,32,32
        x2=x
        #32 32
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
        # 256,16,16
        x3=x
        #16 16
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
        # 512,8,8
        x4=x
        #8 8
        x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        # #4 4
        x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')

        image_feats=[x1,x2,x3,x4,x]

        # point encoder
        # adain
        pc_feats = self.init_pc
        pc_feats = tf.layers.dense(pc_feats,64,activation=tf.nn.relu,use_bias=True)
        pc_feats = tf.layers.dense(pc_feats,64,activation=tf.nn.relu,use_bias=True)
        adain1 = adain(pc_feats,x1)

        pc_feats = tf.layers.dense(pc_feats,128,activation=tf.nn.relu,use_bias=True)
        pc_feats = tf.layers.dense(pc_feats,128,activation=tf.nn.relu,use_bias=True)
        adain2 = adain(pc_feats,x2)

        pc_feats = tf.layers.dense(pc_feats,256,activation=tf.nn.relu,use_bias=True)
        pc_feats = tf.layers.dense(pc_feats,256,activation=tf.nn.relu,use_bias=True)
        adain3 = adain(pc_feats,x3)

        pc_feats = tf.layers.dense(pc_feats,512,activation=tf.nn.relu,use_bias=True)
        pc_feats = tf.layers.dense(pc_feats,512,activation=tf.nn.relu,use_bias=True)
        adain4 = adain(pc_feats,x4)

        pc_feats = tf.layers.dense(pc_feats,512,activation=tf.nn.relu,use_bias=True)
        pc_feats = tf.layers.dense(pc_feats,512,activation=tf.nn.relu,use_bias=True)
        adain5 = adain(pc_feats,x)
        # print(adain1.shape)
        adain1 = tf.layers.dense(adain1,64,activation=None,use_bias=True)
        adain2 = tf.layers.dense(adain2,128,activation=None,use_bias=True)
        adain3 = tf.layers.dense(adain3,256,activation=None,use_bias=True)
        adain4 = tf.layers.dense(adain4,512,activation=None,use_bias=True)
        adain5 = tf.layers.dense(adain5,512,activation=None,use_bias=True)


        # project
        project = get_projection(image_feats, self.init_pc, self.batch_size)

        point_feats = tf.concat([adain1,adain2,adain3,adain4,adain5,project],-1)

        # point decoder
        point = graphX(point_feats, 512, num_instance=500, rank=125, bias=True, activation=tf.nn.relu)
        point = graphX(point, 256, num_instance=1000, rank=250, bias=True, activation=tf.nn.relu)
        point = graphX(point, 128, num_instance=2000, rank=1000, bias=True, activation=tf.nn.relu)
        point = tf.layers.dense(point,3,activation=None,use_bias=True)

        self.gt = self.point
        self.pred = point


def project(img_feats, xs, ys, dim, batch_size):
    out = []
    for i in range(batch_size):
        x,y,img_feat=xs[i],ys[i],img_feats[i,...]
        x1 = tf.floor(x)
        x2 = tf.ceil(x)
        y1 = tf.floor(y)
        y2 = tf.ceil(y)

        Q11 = tf.gather_nd(img_feat, tf.stack([tf.cast(x1,tf.int32), tf.cast(y1,tf.int32)],1))
        Q12 = tf.gather_nd(img_feat, tf.stack([tf.cast(x1,tf.int32), tf.cast(y2,tf.int32)],1))
        Q21 = tf.gather_nd(img_feat, tf.stack([tf.cast(x2,tf.int32), tf.cast(y1,tf.int32)],1))
        Q22 = tf.gather_nd(img_feat, tf.stack([tf.cast(x2,tf.int32), tf.cast(y2,tf.int32)],1))

        weights = tf.multiply(tf.subtract(x2,x), tf.subtract(y2,y))
        Q11 = tf.multiply(tf.tile(tf.reshape(weights,[-1,1]),[1,dim]), Q11)

        weights = tf.multiply(tf.subtract(x,x1), tf.subtract(y2,y))
        Q21 = tf.multiply(tf.tile(tf.reshape(weights,[-1,1]),[1,dim]), Q21)

        weights = tf.multiply(tf.subtract(x2,x), tf.subtract(y,y1))
        Q12 = tf.multiply(tf.tile(tf.reshape(weights,[-1,1]),[1,dim]), Q12)

        weights = tf.multiply(tf.subtract(x,x1), tf.subtract(y,y1))
        Q22 = tf.multiply(tf.tile(tf.reshape(weights,[-1,1]),[1,dim]), Q22)

        out.append(tf.add_n([Q11, Q21, Q12, Q22]))
        # outputs = tf.add_n([Q11, Q21, Q12, Q22])
    out = tf.stack(out)
    # out = tf.transpose(out,perm=[0,2,1])
    return out

def get_projection(img_feat, pc, batch_size):
    X = pc[..., 0]
    Y = pc[..., 1]
    Z = pc[..., 2]

    h = 284 * tf.divide(-Y, -Z) + 128
    w = 284 * tf.divide(X, -Z) + 128

    h = tf.minimum(tf.maximum(h, 0), 255)
    w = tf.minimum(tf.maximum(w, 0), 255)

    x = h/(256.0/64)
    y = w/(256.0/64)
    out1 = project(img_feat[0], x, y, 64, batch_size)

    x = h/(256.0/32)
    y = w/(256.0/32)
    out2 = project(img_feat[1], x, y, 128, batch_size)

    x = h/(256.0/16)
    y = w/(256.0/16)
    out3 = project(img_feat[2], x, y, 256, batch_size)

    x = h/(256.0/8)
    y = w/(256.0/8)
    out4 = project(img_feat[3], x, y, 512, batch_size)

    x = h/(256.0/4)
    y = w/(256.0/4)
    out5 = project(img_feat[4], x, y, 512, batch_size)
    outputs = tf.concat([pc,out1,out2,out3,out4,out5], -1)
    return outputs

def graphX(pc_feats, out_channel, num_instance, rank, bias, activation):
    #main
    main = tf.layers.dense(pc_feats,out_channel,activation=tf.nn.relu,use_bias=True)
    main = graphXConv(main,out_channel,num_instance=num_instance,rank=rank,bias=bias,activation=activation)

    #res
    res = graphXConv(pc_feats,out_channel,num_instance=num_instance,rank=rank,bias=bias,activation=activation)
    return tf.nn.relu(main+res)

def graphXConv(pc_feats, out_channel, num_instance, rank, bias, activation):
    pc_feats = tf.transpose(pc_feats,perm=[0,2,1])
    pc_feats = tf.layers.dense(pc_feats,rank,activation=None,use_bias=False)
    pc_feats = tf.layers.dense(pc_feats,num_instance,activation=None,use_bias=bias)
    pc_feats = tf.transpose(pc_feats,perm=[0,2,1])
    pc_feats = tf.layers.dense(pc_feats,out_channel,activation=activation,use_bias=bias)
    return pc_feats

def adain(pc_feat, img_feat):
    # print(pc_feat.shape)
    # print(img_feat.shape)
    epsilon = 1e-8
    pc_mean, pc_var = tf.nn.moments(pc_feat,[1],keep_dims=True)
    pc_std = tf.sqrt(pc_var+epsilon)
    img_mean, img_var = tf.nn.moments(img_feat,[1,2],keep_dims=True)
    img_std = tf.sqrt(img_var+epsilon)
    pc_feat = (pc_feat-pc_mean)/pc_std

    output = tf.squeeze(img_mean,1) + tf.squeeze(img_std,1) * pc_feat
    # output = (pc_feat+tf.squeeze(img_mean,1)) * tf.squeeze(img_std,1)
    # output = img_std * (pc_feat-pc_mean) / pc_std + img_mean
    return output
