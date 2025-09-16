#!/usr/bin/env python3
"""
Advanced usage example for vid-prepper package.

This example demonstrates more advanced features:
- Batch video processing
- WebDataset integration
- Custom augmentation pipelines
- Performance optimization
"""

import torch
import tempfile
import os
from pathlib import Path
from vid_prepper import metadata, standardize, detector, loader, augmentor


def batch_processing_example():
    """Example of batch video processing."""
    print("Batch Processing Example")
    print("-" * 25)
    
    # Simulate multiple video files
    video_files = ["video1.mp4", "video2.mp4", "video3.mp4"]
    
    # Batch metadata validation
    print("Validating multiple videos...")
    results = metadata.Metadata.validate_videos(
        video_files,
        filters=["filter_missing_video", "filter_resolution", "filter_duration"],
        max_workers=2,
        only_errors=True
    )
    
    print(f"Found {len(results)} videos with errors")
    for result in results:
        print(f"  {result['file']}: {len(result['errors'])} errors")
    
    # Batch standardization
    print("\nStandardizing multiple videos...")
    video_standardizer = standardize.VideoStandardizer(size="224x224", fps=25)
    

    # This would process actual video files
    standardizer.batch_standardize(videos=[video_file_paths], output_dir="videos/")
    


def webdataset_example():
    """Example of WebDataset integration."""
    print("\nWebDataset Integration Example")
    print("-" * 32)
    
    # Create loader for WebDataset
    video_loader = loader.VideoLoader(num_frames=16, size=(224, 224))
    
    # WebDataset path (replace with actual path)
    wds_path = "path/to/webdataset"
    
    print("Creating WebDataset pipeline...")
    # dataset = loader.load_wds(wds_path, key="mp4", label="cls")
    print("WebDataset pipeline created (commented out - requires actual dataset)")
    
    # Standardizer for WebDataset
    video_standardizer = standardize.VideoStandardizer()
    # std_dataset = standardizer.standardize_wds(wds_path)
    print("WebDataset standardization pipeline created")


def custom_augmentation_pipeline():
    """Example of custom augmentation pipeline."""
    print("\nCustom Augmentation Pipeline Example")
    print("-" * 37)
    
    # Create augmentor
    video_augmentor = augmentor.VideoAugmentor(device="cpu")
    
    # Sample video tensor
    video_tensor = torch.randn(2, 16, 3, 224, 224)  # B, T, C, H, W
    
    print(f"Input tensor shape: {video_tensor.shape}")
    
    # Define different augmentation strategies
    strategies = {
        "light_augmentation": [
            ('crop', {'type': 'center', 'size': (200, 200)}),
            ('brightness', {'amount': 0.1})
        ],
        "medium_augmentation": [
            ('crop', {'type': 'random', 'size': (180, 180)}),
            ('flip', {'type': 'horizontal'}),
            ('brightness', {'amount': 0.2}),
            ('contrast', {'amount': 0.1}),
            ('gaussian_blur', {'kernel_size': 3, 'sigma': 0.5})
        ],
        "heavy_augmentation": [
            ('crop', {'type': 'random', 'size': (160, 160)}),
            ('flip', {'type': 'horizontal'}),
            ('brightness', {'amount': 0.3}),
            ('contrast', {'amount': 0.2}),
            ('saturation', {'amount': 0.2}),
            ('gaussian_blur', {'kernel_size': 5, 'sigma': 1.0}),
            ('coarse_dropout', {'number_holes_range': [1, 3]})
        ]
    }
    
    # Apply different strategies
    for strategy_name, augmentations in strategies.items():
        print(f"\nApplying {strategy_name}...")
        result = video_augmentor.chain(video_tensor, augmentations)
        print(f"  Result shape: {result.shape}")


def performance_optimization_example():
    """Example of performance optimization techniques."""
    print("\nPerformance Optimization Example")
    print("-" * 33)
    
    # GPU acceleration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Optimized loader settings
    video_loader = loader.VideoLoader(
        num_frames=16,
        frame_stride=2,
        size=(224, 224),
        device=device,
        use_nvdec=device == "cuda"  # Use NVDEC for GPU acceleration
    )
    
    # Optimized augmentor
    video_augmentor = augmentor.VideoAugmentor(device=device)
    
    # Batch processing for efficiency
    batch_size = 4
    video_tensors = [torch.randn(16, 3, 224, 224) for _ in range(batch_size)]
    batch_tensor = torch.stack(video_tensors)  # B, T, C, H, W
    
    print(f"Processing batch of {batch_size} videos")
    print(f"Batch tensor shape: {batch_tensor.shape}")
    
    # Efficient batch augmentation
    augmented_batch = video_augmentor.chain(batch_tensor, [
        ('crop', {'type': 'random', 'size': (200, 200)}),
        ('flip', {'type': 'horizontal'}),
        ('brightness', {'amount': 0.1})
    ])
    
    print(f"Augmented batch shape: {augmented_batch.shape}")


def main():
    """Main advanced example function."""
    print("vid-prepper Advanced Usage Example")
    print("=" * 40)
    
    try:
        batch_processing_example()
        webdataset_example()
        custom_augmentation_pipeline()
        performance_optimization_example()
        
        print("\n" + "=" * 40)
        print("Advanced example completed successfully!")
        print("\nKey takeaways:")
        print("- Use batch processing for efficiency")
        print("- Leverage WebDataset for large-scale datasets")
        print("- Create custom augmentation pipelines")
        print("- Optimize for your hardware (CPU/GPU)")
        
    except Exception as e:
        print(f"Error during advanced example: {e}")


if __name__ == "__main__":
    main()
