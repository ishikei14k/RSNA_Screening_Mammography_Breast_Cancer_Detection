import argparse
import joblib
import os.path as osp

import mmcv
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ann_file')
    parser.add_argument('result_file')
    parser.add_argument('data_root')
    parser.add_argument('out_dir')
    parser.add_argument('--n-jobs', type=int, default=16)
    return parser.parse_args()


def make_crop(result, filename, data_root, out_dir):
    img = mmcv.imread(osp.join(data_root, filename))
    result = result[0]
    if len(result) > 0:
        crop = mmcv.imcrop(img, result[0, :4])
    else:
        crop = img
    mmcv.imwrite(crop, osp.join(out_dir, filename))


def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.out_dir)
    img_infos = mmcv.load(args.ann_file)['images']
    results = mmcv.load(args.result_file)
    jobs = []
    for img_info, result in zip(img_infos, tqdm(results)):
        job = joblib.delayed(make_crop)(
            result, img_info['file_name'], args.data_root, args.out_dir
        )
        jobs.append(job)

    joblib.Parallel(n_jobs=args.n_jobs, verbose=10)(jobs)


if __name__ == '__main__':
    main()