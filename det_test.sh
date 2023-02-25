exp=$1

export PYTHONPATH=./
python tools/det/test.py data/work_dirs/det/${exp}/${exp}.py \
    data/work_dirs/det/${exp}/latest.pth \
    --out data/work_dirs/det/${exp}/det_result.pkl

python tools/det/postprocess/result_to_csv.py \
    det_ann/train.json \
    data/work_dirs/det/${exp}/det_result.pkl \
    data/work_dirs/det/${exp}/det_result_${exp}.csv