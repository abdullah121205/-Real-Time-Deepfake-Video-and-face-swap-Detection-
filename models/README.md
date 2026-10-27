# Models Directory

This directory contains the neural network architectures, face detection models, and trained model weights used by the real-time detection pipeline.

## Contents

- `face_detection_yunet_2023mar.onnx`: Lightweight (~232 KB) OpenCV YuNet face detector supporting fast multi-face localization and 5-point facial landmarks.
- `*.pth`: PyTorch deepfake classification weights (e.g., `meso4.pth`, `mobilenet_deepfake.pth`).

## Supported Architectures

1. **MesoNet-4 (`meso4`)**:
   - 4-layer mesoscopic convolutional network specifically designed for facial forgery detection.
   - Exceptionally compact (~28,000 parameters), running at 30+ FPS on consumer CPUs.
2. **MesoInception-4 (`meso_inception`)**:
   - Multi-scale inception blocks with dilated convolutions for capturing artifacts across varying spatial resolutions.
3. **MobileNetV3 (`mobilenet`)**:
   - Modern inverted residual bottleneck backbone with custom binary classification head.

## Saving & Loading Weights

To train and export custom weights:
- Open `notebooks/deepfake_training.ipynb` and follow the training pipeline.
- Save the resulting model checkpoint to `models/<model_name>.pth`.
- Run `python app.py --weights models/<model_name>.pth` to use your custom trained checkpoint.
