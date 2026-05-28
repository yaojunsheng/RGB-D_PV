# SolarFusion

Multimodal Deep Learning with Aerial Imagery and nDSM for True-Slope-Aware Rooftop Photovoltaic Potential Estimation.

SolarFusion is a multimodal rooftop PV potential estimation project. It combines RGB aerial imagery and nDSM height data for rooftop parsing, true-slope extraction, PV module placement, and annual electricity generation estimation.

## Project Structure

```text
SolarFusion/
├── FusionPIRNet/              # Main deep learning code for RID1-Depth
│   ├── roof_train_dguide_full.py
│   ├── dataset/
│   ├── model/
│   ├── losses/
│   ├── evaluation/
│   └── data/roof/
├── RID2/FusionPIRNet/         # RID2-Depth version of FusionPIRNet
├── pvcode/                    # Rooftop PV potential estimation pipeline
│   ├── main.py
│   ├── munich_pv_evaluation.py
│   ├── masks_to_vector.py
│   ├── module_placement.py
│   └── electricity_generation.py
├── requirements.txt
└── README.md
```

## Environment

The project is expected to run in the existing conda environment named `region`.

```bash
conda activate region
pip install -r requirements.txt
```

PyTorch should match your local CUDA version. If GPU training fails after installing requirements, reinstall `torch` and `torchvision` from the official PyTorch command for your CUDA driver.

## Data Layout

Dataset download link:

```text
https://1drv.ms/f/c/9108d2bd19b42862/IgCOpCc3sH8MQovOQ7udjt6IAZoEAJqQrhA5-Vx4dobtcD0?e=rRpRNk
```

The training code expects the dataset root to contain a `roof` directory. By default, `FusionPIRNet/roof_train_dguide_full.py` uses `--dataroot ./data`, so run it from inside `FusionPIRNet/`.

Expected layout:

```text
FusionPIRNet/data/roof/
├── train.txt
├── val.txt
├── test.txt
├── images/        # RGB aerial images
├── seg6/          # 6-class roof labels
├── seg9/          # 9-class roof labels
└── seg_height/    # nDSM height files, usually .tif
```

File names in `train.txt`, `val.txt`, and `test.txt` should match the image, label, and height file names.

## Training

Run RID1-Depth training:

```bash
conda activate region
cd FusionPIRNet
python roof_train_dguide_full.py \
  --dataroot ./data \
  --task-type seg6 \
  --single-gpu-id 0
```

Run RID2-Depth training:

```bash
conda activate region
cd RID2/FusionPIRNet
python roof_train_dguide_full.py \
  --dataroot ./data \
  --task-type seg6 \
  --single-gpu-id 0
```

Training outputs are saved under `results/ablation_study/` by default.

## PV Potential Estimation

The PV estimation scripts are in `pvcode/`. Before running, edit the path configuration at the top of the target script, such as:

- `dir_roof_segment_masks`
- `dir_roof_superstructure_masks`
- `dir_geotifs`
- `dir_ndsm`
- `dir_pvgis_cache`
- `dir_results`
- `PIXEL_SIZE`

Then run:

```bash
conda activate region
cd pvcode
python main.py
```

For the Munich city-scale evaluation:

```bash
conda activate region
cd pvcode
python munich_pv_evaluation.py
```

## Notes

- Keep the original directory structure unchanged, because several scripts use relative imports and fixed data paths.
- Large datasets, model checkpoints, generated masks, GeoTIFF files, and PVGIS cache files are not included in this README.
- Some PV estimation scripts call the PVGIS API through `requests`; internet access is required when cached radiation data are unavailable.

## Citations

> Junsheng Yao, Sebastian Krapf, Qingyu Li. Multimodal Deep Learning with Aerial Imagery and nDSM for True-Slope-Aware Rooftop Photovoltaic Potential Estimation. ISPRS Journal of Photogrammetry and Remote Sensing, 2026.
