import os
import requests
from tqdm import tqdm


def download_file(url, file_path):

    if not os.path.isfile(file_path):
        print('Downloading file', file_path)
        chunk_size = 1024
        r = requests.get(url, stream=True)
        total_size = int(r.headers['content-length'])
        wrote = 0
        with open(file_path, 'wb') as f:
            for data in tqdm(iterable=r.iter_content(chunk_size=chunk_size),
                             total=total_size/chunk_size, unit='KB'):
                wrote += len(data)
                f.write(data)
        if total_size != 0:
            print("Check the file!")
        else:
            print('{} download Complete!'.format(file_path))


def create_folders(folders_list):

    for folder in folders_list:
        if not os.path.isdir(folder):
            print('Creating folder', folder)
            os.mkdir(folder)


def prepare_repo():

    # Make folders
    folders_list = ['weights']
    create_folders(folders_list)

    # Download VGG weights
    vgg_url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    vgg_weights_path = 'weights/vgg16.h5'
    download_file(vgg_url, vgg_weights_path)

    # Download weights for all networks
