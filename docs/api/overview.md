# API Documentation

## Core Modules

### Metadata Module
The `metadata` module provides video metadata extraction and validation capabilities.

#### Classes

##### `Metadata`
Main class for extracting and validating video metadata using FFprobe.

**Methods:**
- `run()` - Extract metadata from video file
- `filter_missing_video()` - Check if video has video stream
- `filter_missing_audio()` - Check if video has audio stream
- `filter_resolution(min_width, min_height)` - Validate video resolution
- `filter_duration(min_seconds)` - Validate video duration
- `filter_codecs(allowed)` - Validate video codecs
- `export_errors()` - Export validation errors

**Static Methods:**
- `validate_videos(inputs, filters, max_workers, only_errors)` - Batch validation

### Augmentor Module
The `augmentor` module provides video augmentation capabilities for deep learning.

#### Classes

##### `VideoAugmentor`
Efficient video augmentation class for tensors.

**Methods:**
- `crop(videos, type, size)` - Crop video frames
- `flip(videos, type)` - Flip video frames
- `mirror(videos, edge)` - Mirror video frames
- `pad(videos, proportion, fill)` - Pad video frames
- `brightness(videos, amount)` - Adjust brightness
- `contrast(videos, amount)` - Adjust contrast
- `saturation(videos, amount)` - Adjust saturation
- `gaussian_blur(videos, kernel_size, sigma)` - Apply Gaussian blur
- `chain(videos, augmentations)` - Apply multiple augmentations

### Detector Module
The `detector` module provides video analysis capabilities.

#### Classes

##### `VideoDetector`
Video analysis class for shots, wipes, and object detection.

**Methods:**
- `detect_shots(video_path, method)` - Detect shot boundaries
- `detect_wipes(video_tensor, block_grid, threshold)` - Detect wipe transitions
- `detect_objects(video_tensor, classes, conf_thresh)` - Detect objects using YOLO

### Loader Module
The `loader` module provides video loading and tensor conversion.

#### Classes

##### `VideoLoader`
Loads videos as PyTorch tensors with GPU acceleration support.

**Methods:**
- `load_file(filepath)` - Load single video file
- `load_bytes(video_bytes)` - Load video from bytes
- `load_files(filepaths)` - Load multiple videos as batch
- `load_wds(wds_path, key, label)` - Load WebDataset

### Standardize Module
The `standardize` module provides video standardization capabilities.

#### Classes

##### `VideoStandardizer`
Standardizes videos for deep learning models.

**Methods:**
- `standardize_video(video_input)` - Standardize single video
- `batch_standardize(videos, out_dir)` - Standardize multiple videos
- `standardize_wds(wds_path, key, label)` - Standardize WebDataset
