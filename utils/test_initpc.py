import numpy as np

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

print(get_batch_init_pc(32,250).shape)
