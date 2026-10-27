# Data Directory

This directory stores datasets for training, validating, and testing deepfake and face-swap detection models.

## Recommended Directory Structure

```text
data/
├── README.md
├── raw/
│   ├── real/        <- Genuine / authentic video clips or frames
│   └── fake/        <- Manipulated / deepfake / face-swapped video clips or frames
└── processed/
    ├── train/
    │   ├── real/    <- Cropped, normalized 224x224 authentic faces
    │   └── fake/    <- Cropped, normalized 224x224 manipulated faces
    ├── val/
    │   ├── real/
    │   └── fake/
    └── test/
        ├── real/
        └── fake/
```

## Supported Datasets

1. **FaceForensics++ (FF++)**:
   - Includes Deepfakes, Face2Face, FaceSwap, and NeuralTextures with raw, c23, and c40 compression levels.
   - [Paper & Code](https://github.com/ondyari/FaceForensics)
2. **Celeb-DF (v2)**:
   - High-quality deepfake video dataset with reduced visual artifacts compared to earlier datasets.
   - [Website](https://github.com/yuezunli/celeb-deepfakeforensics)
3. **Deepfake Detection Challenge (DFDC)**:
   - Diverse dataset released by Meta / AWS with varied lighting, head poses, and face-swapping algorithms.

> **Note**: Raw video files and high-resolution datasets should not be committed to Git. Ensure they remain in `data/raw/` or `data/processed/`, which are automatically ignored by `.gitignore`.
