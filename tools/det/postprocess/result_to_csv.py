import argparse
import joblib
import os.path as osp
import numpy as np
import pandas as pd

import mmcv
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ann_file')
    parser.add_argument('result_file')
    parser.add_argument('out_path')
    return parser.parse_args()


def main():
    args = parse_args()
    img_infos = mmcv.load(args.ann_file)['images']
    results = mmcv.load(args.result_file)
    
    name_list = []
    x_list = []
    y_list = []
    w_list = []
    h_list = []
    score_list = []
    for img_info, result in tqdm(zip(img_infos, tqdm(results)), total=len(img_infos)):
        name_list.append(img_info['file_name'])
        result = result[0]
        if len(result) > 0:
            x_list.append(np.max([0, int(result[0, 0])]))
            y_list.append(np.max([0, int(result[0, 1])]))
            w_list.append(np.max([0, int(result[0, 2])]))
            h_list.append(np.max([0, int(result[0, 3])]))
            score_list.append(result[0, 4])
        else:
            x_list.append(np.nan)
            y_list.append(np.nan)
            w_list.append(np.nan)
            h_list.append(np.nan)
            score_list.append(np.nan)

    df = pd.DataFrame({
        'name': name_list,
        'xmin': y_list,
        'ymin': x_list,
        'xmax': h_list,
        'ymax': w_list,
        'score': score_list
    })

    df.to_csv(args.out_path, index=False)
        

if __name__ == '__main__':
    main()