exp=$1

export PYTHONPATH=./
python tools/det/train.py config/det/${exp}.py \
    --work-dir data/work_dirs/det/${exp} \
    --seed 26268661