""" modified from https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import os
import lmdb # install lmdb by "pip install lmdb"
import cv2 as cv
import fire
import numpy as np

def check_image_is_valid(image_bin):
    if image_bin is None: return False
    image_buf = np.fromstring(image_bin, dtype=np.uint8)
    img = cv.imdecode(image_buf, cv.IMREAD_GRAYSCALE)
    im_height, im_width = img.shape[0], img.shape[1]
    if im_height * im_width == 0:
        return False
    return True
        
            
def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)

def create_dataset(output_path: str, image_path_list: list, label_list: list, lexicon_list: list = None, check_valid: bool = True):
    """
    Create LMDB dataset for CRNN training.
    Args:
        output_path (str): LMDB output path
        image_path_list (list): list of image path
        label_list (list): list of corresponding groundtruth texts
        lexicon_list (list, optional): (optional) list of lexicon lists
        check_valid (bool, optional): if true, check the validity of every image
    """
    
    assert(len(image_path_list)) == len(label_list)
    nsample = len(image_path_list)
    env = lmdb.open(output_path, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nsample):
        image_path = image_path_list[i]
        label = label_list[i]
        if not os.path.exists(image_path):
            print(f'{image_path} does not exist')
            continue
        with open(image_path, 'r') as f:
            image_bin = f.read()
        if check_valid:
            if not check_image_is_valid(image_bin):
                print(f'{image_path} is not valid image')
                continue
        
        image_key = f'image-{cnt:09d}'
        label_key = f'label-{cnt:09d}'
        cache[image_key] = image_bin
        cache[label_key] = label
        if lexicon_list:
            lexicon_key = f'lexicon-{cnt:09d}'
            cache[lexicon_key] = ' '.join(lexicon_list[i])
        if cnt % 1000 == 0:
            write_cache(env, cache)
            cache = {}
            print(f'Written {cnt} / {nsample}')
        cnt = cnt + 1
    nsample = cnt-1
    cache['num-samples'] = str(nsample)
    write_cache(env, cache)
    print(f'Created dataset with {nsample} samples')
    # print('Created dataset with %d samples' % nSamples)
    

if __name__ == '__main__':
    fire.Fire(create_dataset)
