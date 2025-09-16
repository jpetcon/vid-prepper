import pytest
import torch
from vid_prepper.augmentor import VideoAugmentor


class TestVideoAugmentor:
    """Test cases for the VideoAugmentor class."""
    
    @pytest.fixture
    def augmentor(self):
        """Create VideoAugmentor instance for testing."""
        return VideoAugmentor(device="cpu")
    
    @pytest.fixture
    def sample_video(self):
        """Create sample video tensor for testing."""
        return torch.randn(16, 3, 224, 224)  # T, C, H, W
    
    @pytest.fixture
    def sample_batch(self):
        """Create sample batch tensor for testing."""
        return torch.randn(2, 16, 3, 224, 224)  # B, T, C, H, W
    
    def test_init(self):
        """Test VideoAugmentor initialization."""
        augmentor = VideoAugmentor(device="cpu")
        assert augmentor.device == "cpu"
        
        augmentor_cuda = VideoAugmentor(device="cuda")
        assert augmentor_cuda.device == "cuda"
    
    def test_merge_batch_time(self, augmentor, sample_video, sample_batch):
        """Test batch-time dimension merging."""
        # Single video
        merged = augmentor._merge_batch_time(sample_video)
        assert merged.shape == (16, 3, 224, 224)
        
        # Batch video
        merged = augmentor._merge_batch_time(sample_batch)
        assert merged.shape == (32, 3, 224, 224)  # B*T, C, H, W
    
    def test_unmerge_batch_time(self, augmentor, sample_video, sample_batch):
        """Test batch-time dimension unmerging."""
        # Single video
        merged = augmentor._merge_batch_time(sample_video)
        unmerged = augmentor._unmerge_batch_time(merged, sample_video.shape)
        assert unmerged.shape == sample_video.shape
        
        # Batch video
        merged = augmentor._merge_batch_time(sample_batch)
        unmerged = augmentor._unmerge_batch_time(merged, sample_batch.shape)
        assert unmerged.shape == sample_batch.shape
    
    def test_crop_random(self, augmentor, sample_video):
        """Test random cropping."""
        cropped = augmentor.crop(sample_video, type="random", size=(100, 100))
        assert cropped.shape == (16, 3, 100, 100)
    
    def test_crop_center(self, augmentor, sample_video):
        """Test center cropping."""
        cropped = augmentor.crop(sample_video, type="center", size=(100, 100))
        assert cropped.shape == (16, 3, 100, 100)
    
    def test_crop_invalid_type(self, augmentor, sample_video):
        """Test invalid crop type."""
        with pytest.raises(ValueError, match="crop type must be 'random' or 'center'"):
            augmentor.crop(sample_video, type="invalid")
    
    def test_flip_horizontal(self, augmentor, sample_video):
        """Test horizontal flipping."""
        flipped = augmentor.flip(sample_video, type="horizontal")
        assert flipped.shape == sample_video.shape
        # Check that flipping actually occurred
        assert not torch.equal(flipped, sample_video)
    
    def test_flip_vertical(self, augmentor, sample_video):
        """Test vertical flipping."""
        flipped = augmentor.flip(sample_video, type="vertical")
        assert flipped.shape == sample_video.shape
    
    def test_flip_invalid_type(self, augmentor, sample_video):
        """Test invalid flip type."""
        with pytest.raises(ValueError, match="flip type must be 'horizontal' or 'vertical'"):
            augmentor.flip(sample_video, type="invalid")
    
    def test_pad(self, augmentor, sample_video):
        """Test padding."""
        padded = augmentor.pad(sample_video, proportion="10%")
        assert padded.shape[0] == sample_video.shape[0]  # T dimension unchanged
        assert padded.shape[1] == sample_video.shape[1]  # C dimension unchanged
        assert padded.shape[2] > sample_video.shape[2]   # H dimension increased
        assert padded.shape[3] > sample_video.shape[3]   # W dimension increased
    
    def test_brightness(self, augmentor, sample_video):
        """Test brightness adjustment."""
        brightened = augmentor.brightness(sample_video, amount=0.2)
        assert brightened.shape == sample_video.shape
    
    def test_contrast(self, augmentor, sample_video):
        """Test contrast adjustment."""
        contrasted = augmentor.contrast(sample_video, amount=0.2)
        assert contrasted.shape == sample_video.shape
    
    def test_saturation(self, augmentor, sample_video):
        """Test saturation adjustment."""
        saturated = augmentor.saturation(sample_video, amount=0.2)
        assert saturated.shape == sample_video.shape
    
    def test_color_adjust(self, augmentor, sample_video):
        """Test color adjustment."""
        adjusted = augmentor.color_adjust(sample_video, red=1.2, green=0.8, blue=1.1)
        assert adjusted.shape == sample_video.shape
    
    def test_gaussian_blur(self, augmentor, sample_video):
        """Test Gaussian blur."""
        blurred = augmentor.gaussian_blur(sample_video, kernel_size=5, sigma=1.0)
        assert blurred.shape == sample_video.shape
    
    def test_coarse_dropout(self, augmentor, sample_video):
        """Test coarse dropout."""
        dropped = augmentor.coarse_dropout(sample_video, number_holes_range=[1, 2])
        assert dropped.shape == sample_video.shape
    
    def test_chain(self, augmentor, sample_video):
        """Test augmentation chaining."""
        augmentations = [
            ('crop', {'type': 'center', 'size': (100, 100)}),
            ('flip', {'type': 'horizontal'}),
            ('brightness', {'amount': 0.1})
        ]
        
        result = augmentor.chain(sample_video, augmentations)
        assert result.shape == (16, 3, 100, 100)
    
    def test_chain_invalid_augmentation(self, augmentor, sample_video):
        """Test chaining with invalid augmentation."""
        augmentations = [('invalid_aug', {})]
        
        with pytest.raises(ValueError, match="Unknown augmentation"):
            augmentor.chain(sample_video, augmentations)
