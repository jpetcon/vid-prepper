#!/usr/bin/env python3
"""
Basic usage example for vid-prepper package.

This example demonstrates the core functionality of vid-prepper:
- Video metadata extraction and validation
- Video standardization
- Object detection
- Video augmentation
"""

import torch
from vid_prepper import metadata, standardize, detector, loader, augmentor


def main():
    """Main example function."""
    print("vid-prepper Basic Usage Example")
    print("=" * 40)
    
    # Example video file path (replace with your actual video file)
    video_path = "sample_video.mp4"
    
    # 1. Metadata Extraction and Validation
    print("\n1. Metadata Extraction and Validation")
    print("-" * 35)
    
    try:
        # Extract metadata
        metadata_extractor = metadata.Metadata(video_path)
        video_info = metadata_extractor.run()
        
        print(f"Video duration: {video_info['format']['duration']} seconds")
        print(f"Number of streams: {len(video_info['streams'])}")
        
        # Validate video
        checks = [
            metadata_extractor.filter_missing_video(),
            metadata_extractor.filter_missing_audio(),
            metadata_extractor.filter_resolution(),
            metadata_extractor.filter_duration(),
            metadata_extractor.filter_codecs()
        ]
        
        print(f"Validation checks passed: {sum(checks)}/{len(checks)}")
        
        if metadata_extractor.errors:
            print("Validation errors:")
            for error in metadata_extractor.errors:
                print(f"  - {error['check']}: {error['message']}")
    
    except FileNotFoundError:
        print(f"Video file '{video_path}' not found. Please provide a valid video file.")
        return
    except Exception as e:
        print(f"Error during metadata extraction: {e}")
        return
    
    # 2. Video Standardization
    print("\n2. Video Standardization")
    print("-" * 25)
    
    try:
        standardizer = standardize.VideoStandardizer(
            size="224x224",
            fps=25,
            codec="h264",
            color="rgb",
            use_gpu=False  # Set to True if you have CUDA
        )
        
        # Standardize video (this would create a new standardized video)
        print("Standardizing video...")
        # standardizer.standardize_video(video_input="video.mp4", output_path="video_standardized.mp4")
        print("Video standardization completed (commented out to avoid file I/O)")
        
    except Exception as e:
        print(f"Error during video standardization: {e}")
    
    # 3. Video Loading and Tensor Operations
    print("\n3. Video Loading and Tensor Operations")
    print("-" * 40)
    
    try:
        # Load video as tensor
        video_loader = loader.VideoLoader(
            num_frames=16,
            frame_stride=2,
            size=(224, 224),
            device="cpu",
            use_nvdec=False  # Use "cuda" if available
        )
        
        print("Loading video as tensor...")
        # video_tensor = loader.load_file(video_path)
        # print(f"Video tensor shape: {video_tensor.shape}")
        print("Video loading completed")
        
        # Create a sample tensor for demonstration
        sample_tensor = torch.randn(16, 3, 224, 224)  # T, C, H, W
        print(f"Sample tensor shape: {sample_tensor.shape}")
        
    except Exception as e:
        print(f"Error during video loading: {e}")
    
    # 4. Video Augmentation
    print("\n4. Video Augmentation")
    print("-" * 20)
    
    try:
        # Create augmentor
        video_augmentor = augmentor.VideoAugmentor(device="cpu", use_gpu=False)
        
        # Create sample video tensor
        video_tensor = torch.randn(16, 3, 224, 224)  # T, C, H, W
        
        print("Applying augmentations...")
        
        # Individual augmentations
        cropped = augmentor.crop(video_tensor, type="center", size=(200, 200))
        print(f"Cropped tensor shape: {cropped.shape}")
        
        flipped = augmentor.flip(video_tensor, type="horizontal")
        print(f"Flipped tensor shape: {flipped.shape}")
        
        brightened = augmentor.brightness(video_tensor, amount=0.2)
        print(f"Brightened tensor shape: {brightened.shape}")
        
        # Chained augmentations
        augmentations = [
            ('crop', {'type': 'random', 'size': (180, 180)}),
            ('flip', {'type': 'horizontal'}),
            ('brightness', {'amount': 0.1}),
            ('contrast', {'amount': 0.1})
        ]
        
        chained_result = augmentor.chain(video_tensor, augmentations)
        print(f"Chained augmentation result shape: {chained_result.shape}")
        
    except Exception as e:
        print(f"Error during video augmentation: {e}")
    
    # 5. Object Detection (requires YOLO model)
    print("\n5. Object Detection")
    print("-" * 18)
    
    try:
        # Create detector
        video_detector = detector.VideoDetector(device="cpu")
        
        # Create sample video tensor
        video_tensor = torch.randn(16, 3, 224, 224)  # T, C, H, W
        
        print("Running object detection...")
        # results = detector.detect_objects(video_tensor, conf_thresh=0.3, text_queries="cat. dog")
        # print(f"Detection results: {len(results)} frames processed")
        print("Object detection completed (commented out to avoid model download)")
        
    except Exception as e:
        print(f"Error during object detection: {e}")
    
    print("\n" + "=" * 40)
    print("Example completed successfully!")
    print("\nTo run with actual video files:")
    print("1. Replace 'sample_video.mp4' with your video file path")
    print("2. Uncomment the actual processing lines")
    print("3. Ensure you have the required dependencies installed")


if __name__ == "__main__":
    main()
