# RSNA Screening Mammography Breast Cancer Detection
## Environment
```bash
docker build -t ${image_name} -f docker/Dockerfile . --no-cache
docker run --gpus all -itd -v ${working_dir}:/home/working --shm-size=128g --name ${container_name} -h host ${image_name} /bin/bash
docker exec -it ${container_name} /bin/bash
```

## How to use
1. Download data from kaggle and unzip all to `./data`
```bash
kaggle competitions download -c rsna-breast-cancer-detection
```

2. Prepare dataset.
```bash
python tools/det/preprocess/prepare_dataset.py
```

3. Train detector. (with 1,000 annotated train images)
```bash
sh det_train.sh 001_baseline
```

4. Inference all of the train images and Convert results to csv.
```bash
sh det_test.sh 001_baseline
```
Bbox will be saved to: `data/work_dirs/det/001_baseline/det_result_001_baseline.csv`