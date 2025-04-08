# Towards Unconstrained 2D Pose Estimation of the Human Spine

<div align="center">

[![Conference](https://img.shields.io/badge/CVPRW-2025-6b8bc7.svg?style=for-the-badge)](https://cvpr2025.thecvf.com/)
[![PyPI version](https://img.shields.io/pypi/v/spinepose.svg?style=for-the-badge)](https://pypi.org/project/spinepose/)
[![License](https://img.shields.io/badge/License-CC--BY--NC--4.0-lightgrey.svg?style=for-the-badge)](LICENSE)

![](data/demo/outputs/video1.gif)
![](data/demo/outputs/video2.gif)
</div>

---

## Contents
- [Abstract](#abstract)
- [Installation](#installation)
- [Inference](#inference)
- [SpineTrack Dataset](#spinetrack-dataset)
- [Training and Evaluation](#training-and-evaluation)
- [Citation](#citation)
- [License](#license)

---

## Abstract
We introduce SpineTrack, the first comprehensive dataset dedicated to 2D spine pose estimation in unconstrained environments, addressing a critical gap in human pose analysis for sports and biomechanical applications. Existing pose datasets typically represent the spine with a single rigid segment, neglecting the detailed articulation required for precise analysis. To overcome this limitation, SpineTrack comprises two complementary components: SpineTrack-Real, a real-world dataset with high-fidelity spine annotations refined via an active learning pipeline, and SpineTrack-Unreal, a synthetic dataset generated using an Unreal Engine-based framework with accurate ground-truth labels. Additionally, we propose a novel biomechanical validation framework based on OpenSim to enforce anatomical consistency in the annotated keypoints. Complementing the dataset, our SpinePose model extends state-of-the-art body pose estimation networks through a teacher–student distillation approach and an anatomical regularization strategy, effectively incorporating detailed spine keypoints without sacrificing overall performance. Extensive experiments on standard benchmarks and sports-specific scenarios demonstrate that our approach significantly improves spine tracking accuracy while maintaining robust generalization.

---

## Installation
**Recommended Python Version:** 3.9–3.12

```bash
pip install spinepose
```

On Linux/Windows with CUDA available, install the GPU version:

```bash
pip install spinepose[gpu]
```

> [!NOTE]
> For model training or reproducing the full pipeline, please refer to the [Training and Evaluation](#training-and-evaluation) section.

## Inference

The `spinepose` package provides a command-line interface and a Python API for quick spinal keypoint predictions on images and videos.

### Using the CLI

```bash
spinepose -i /path/to/image_or_video -o /path/to/output
```

This automatically downloads the model weights (if not already present) and outputs the annotated image or video. Use spinepose -h to view all available options, including GPU usage and confidence thresholds.

### Using the Python API

```python
import cv2
from spinepose import SpinePoseEstimator

# Initialize estimator (downloads ONNX model if not found locally)
estimator = SpinePoseEstimator(device='cuda')

# Perform inference on a single image
image = cv2.imread('path/to/image.jpg')
keypoints, scores = estimator.predict(image)
visualized = estimator.visualize(image, keypoints, scores)
cv2.imwrite('output.jpg', visualized)
```

Or, for a simplified interface:

```python
from spinepose.inference import infer_image, infer_video

# Single image inference
infer_image('path/to/image.jpg', 'output.jpg')

# Video inference with optional temporal smoothing
infer_video('path/to/video.mp4', 'output_video.mp4', use_smoothing=True)
```

## SpineTrack Dataset

> [!NOTE]
> Detailed dataset documentation will be added soon, including download links, annotation structure, and usage guidelines.

## Training and Evaluation

> [!NOTE]
> Step-by-step guide and scripts for reproducing our training pipelines, baseline models, and evaluation metrics will be provided in the future.

---

## Citation

If this project or dataset proves helpful in your work, please cite:

```bibtex
@inproceedings{khan2025cvprw,
    author    = {Khan, Muhammad Saif Ullah and Krauß, Stephan and Stricker, Didier},
    title     = {Towards Unconstrained 2D Pose Estimation of the Human Spine},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {},
    year      = {2025},
    pages     = {}
}
```

## License

This project is released under the [CC-BY-NC-4.0 License](LICENSE). Commercial use is prohibited, and appropriate attribution is required for research or educational applications.
