# vid-prepper

Usage examples available at: https://github.com/jpetcon/vid-prepper/tree/main/examples

Package for preparing videos for deep learning models. Built on the giant shoulders of FFMPEG, NVIDIA DALI, OpenCV, Kornia, PySceneDetect, Torchvision and PyTorch.

This package attempts to bring some common video pre-processing methods together in an efficient way for both CPU and GPU.

## Installation

### Prerequisites

Before installing vid-prepper, you need to install the following external dependencies:

#### FFmpeg (Required)
FFmpeg is required for video processing operations.

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS (with Homebrew):**
```bash
brew install ffmpeg
```

**Windows:**
Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) and add to PATH.

#### CUDA/NVDEC (Optional but Recommended)
For GPU acceleration, install CUDA toolkit and ensure your system supports NVDEC:

**Ubuntu/Debian:**
```bash
# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit
```

**macOS:**
CUDA is not supported on macOS. The package will automatically fall back to CPU processing.

**Windows:**
Download CUDA toolkit from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

### Package Installation

```bash
pip install vid-prepper
```

## Usage

```python
from vid_prepper import metadata, standardize

# Example usage
metadata_extractor = metadata.Metadata("video.mp4")
video_info = metadata_extractor.run()

standardizer = standardize.VideoStandardizer(
        size="224x224",
        fps=25,
        codec="h264",
        color="rgb",
        use_gpu=False  # Set to True if you have CUDA
    )

standardizer.standardize_video(video_input="video.mp4", output_path="video_standardized.mp4")
```

## Features

- Video metadata extraction and validation
- Video standardization and preprocessing
- Object detection and scene analysis
- Video augmentation for deep learning
- Efficient tensor loading with GPU acceleration
