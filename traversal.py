import numpy as np
from data_process import create_dir, generate_latent_points, prediction_post_process
from PIL import Image
import numpy as np
import os
import imageio

import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def genGif(images_dir, save_dir):
    images = []
    for file in sorted_alphanumeric(os.listdir(images_dir)):
        if '.png' in file:
            images.append(imageio.imread(f'{images_dir}/{file}'))
    imageio.mimsave(f'{save_dir}/gif.gif', images)       

def getTransitionPoints(pt1, pt2, n_steps):
    n_steps -= 1
    dist = np.linalg.norm(pt2 - pt1) # euclidean distance between p1 and p2
    dhat = (pt2 - pt1) / dist # unit vector that defines the line from p1 to p2
    pts = [pt1]
    for i in range(1, n_steps + 1):
        pt = pt1 + i*(dist / n_steps)*dhat
        pts.append(pt)
    return np.array(pts)

def gen_traversal_points(latent_dim, n_samples, n_steps, latent_points):
    if latent_points is None:
        latent_points = generate_latent_points(latent_dim, n_samples)
    pts = []
    for i in range(0, n_samples - 1):
        new = getTransitionPoints(latent_points[i], latent_points[i + 1], n_steps).tolist()
        if i != 0:
            new = new[1:]
        pts += new
    return pts, latent_points

def traverse_latent_space(g_model, latent_dim, save_dir, n_samples=10, n_steps=20, batch_size=10, latent_points=None):
    # devise name
    name = create_dir(save_dir, g_model)
    pts, latent_points = gen_traversal_points(latent_dim, n_samples, n_steps, latent_points)

    file_names = []
    for i in range(0, len(pts), batch_size):
        batch = pts[i:i + batch_size]
        X = g_model.predict(batch)
        prediction_post_process(X, f'{save_dir}/{name}/plot_{i}', i)
        file_names += [f'{save_dir}/{name}/plot_{str(i)}_{str(j)}_{i + j}.png' for j in range(len(X))]
        print(f'Generated Batch {i/10 + 1} out of {len(pts) // batch_size}')

    print(f'traversal samples generated at {save_dir}')
    return file_names, np.array(pts)

def read_latent_points(file_path, latent_dim, n_samples):
    dims = np.loadtxt(file_path).reshape(n_samples, latent_dim)
    return dims

# path = 'E:\good_dims_generator_256x256-tuned-95352-7\dims_512_26_3.txt'
# dims_1 = np.loadtxt(path).reshape(26, 512).tolist()
# path = 'E:\good_dims_generator_256x256-tuned-95352-7\dims_512_36_4.txt'
# dims_2 = np.loadtxt(path).reshape(36, 512).tolist()

# dims = np.array(dims_1 + dims_2)
# with open(f'E:\good_dims_generator_256x256-tuned-95352-7/dims_{512}_{len(dims)}_combined.txt', 'w') as f:
#     for row in dims:
#         np.savetxt(f, row)
# good_ones_3 = [1151, 1683, 1451, 1812, 624, 1627, 1779, 1100, 1711, 1423, 475, 1234, 1607, 585, 1834, 1181, 1125, 1735, 85, 1572, 1465, 1868, 1770, 1352, 1486, 1441, 757, 1139, 1881, 468, 518, 1478, 613, 1540,  1174, 997]
# good_ones_pts = np.array([dims[x] for x in good_ones_3])
# with open(f'E:\good_dims_generator_256x256-tuned-95352-7/dims_{512}_{len(good_ones_pts)}_4.txt', 'w') as f:
#     for row in good_ones_pts:
#         np.savetxt(f, row)

# 'E:\good_dims_generator_256x256-tuned-95352-7\dims_512_62_combined.txt'
genGif('E:/traversal_good_ones/256x256', 'E:/traversal_good_ones/')