# SNU_fastMRI2021

Repository for source code used in SNU fastMRI challenge 2021.
(Using undersampled MRI data (both k-space and image domain) to reconstruct original image)

## Usage

```bash
cd Code

python3 train.py
python3 evaluate.py
python3 leaderboard_eval.py
```

## Aditional notes
train.py takes ~15 hours (for 35 epochs) (on NVIDIA GeForce RTX 3090) - it will make weight checkpoints for the model

evaluate.py will create reconstructed images for data in 'image_Leaderboard'

leaderboard_eval.py will compute the average SSIM for your reconstructed images
