# Getting Started Tutorial

This tutorial will guide you through the basic usage of vid-prepper.

## Installation

First, install the required external dependencies:

### FFmpeg (Required)
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### CUDA (Optional)
For GPU acceleration:
```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit
```

### Python Package
```bash
pip install vid-prepper
# Or with GPU support
pip install vid-prepper[gpu]
```

## Basic Usage

### 1. Video Metadata Extraction

```python
from vid_prepper import metadata

# Extract metadata
extractor = metadata.Metadata("your_video.mp4")
video_info = extractor.run()

print(f"Duration: {video_info['format']['duration']} seconds")
print(f"Resolution: {video_info['streams'][0]['width']}x{video_info['streams'][0]['height']}")
```

### 2. Video Validation

```python
# Validate video quality
checks = [
    extractor.filter_missing_video(),
    extractor.filter_resolution(min_width=320, min_height=240),
    extractor.filter_duration(min_seconds=1.0),
    extractor.filter_codecs(allowed=["h264", "hevc"])
]

print(f"Validation passed: {sum(checks)}/{len(checks)}")

# Check for errors
if extractor.errors:
    for error in extractor.errors:
        print(f"Error: {error['check']} - {error['message']}")
```

### 3. Video Loading

```python
from vid_prepper import tensor_load

# Load video as tensor
loader = tensor_load.VideoLoader(
    num_frames=16,
    frame_stride=2,
    size=(224, 224),
    device="cuda"  # or "cpu"
)

video_tensor = loader.load_file("your_video.mp4")
print(f"Video shape: {video_tensor.shape}")  # (T, C, H, W)
```

### 4. Video Augmentation

```python
from vid_prepper import augment

# Create augmentor
augmentor = augment.VideoAugmentor(device="cuda")

# Apply augmentations
augmented = augmentor.chain(video_tensor, [
    ('crop', {'type': 'random', 'size': (200, 200)}),
    ('flip', {'type': 'horizontal'}),
    ('brightness', {'amount': 0.1})
])

print(f"Augmented shape: {augmented.shape}")
```

### 5. Object Detection

```python
from vid_prepper import detect

# Create detector
detector = detect.VideoDetector(device="cuda")

# Detect objects
results = detector.detect_objects(video_tensor, conf_thresh=0.3)

for frame_idx, frame_results in enumerate(results):
    print(f"Frame {frame_idx}: {len(frame_results['boxes'])} objects detected")
```

## Next Steps

- Check out the [Advanced Usage](advanced_usage.md) tutorial
- Explore the [API Documentation](api/overview.md)
- Run the [examples](../examples/) to see more use cases
