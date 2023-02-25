import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import os
import pydicom
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from pydicom.pixel_data_handlers.util import apply_voi_lut


DATA_DIR = Path('data/')
OUT_DIR = Path('data/preprocessed/')
IMG_SIZE = 2048
VOI_LUT = True
EXPORT_DIR = OUT_DIR/f'image_resized_{IMG_SIZE}{"V" if VOI_LUT else ""}'
os.makedirs(EXPORT_DIR, exist_ok=True)
print(os.listdir(OUT_DIR))
N_JOBS = 8


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def process(f, size=1024):
    patient_id = f.parent.name
    if not (EXPORT_DIR/patient_id).exists():
        (EXPORT_DIR/patient_id).mkdir(exist_ok=True)
    image_id = f.stem

    dicom = pydicom.dcmread(f)
    img = dicom.pixel_array

    if VOI_LUT:
        img = apply_voi_lut(img, dicom)

    if dicom.PhotometricInterpretation == "MONOCHROME1":
        img = img.max() - img

    img = (img - img.min()) / (img.max() - img.min())
    img = cv2.resize(img, (size, size))
    out_path = str(EXPORT_DIR/f'{patient_id}/{image_id}.png')
    cv2.imwrite(out_path, (img * 255).astype(np.uint8))


train_images = list((DATA_DIR/'train_images/').glob('**/*.dcm'))
_ = ProgressParallel(n_jobs=N_JOBS)(
    delayed(process)(img_path, size=IMG_SIZE) for img_path in tqdm(train_images)
)