import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from vid_prepper.metadata import Metadata, MetadataError


class TestMetadata:
    """Test cases for the Metadata class."""
    
    def test_init(self):
        """Test Metadata initialization."""
        metadata = Metadata("test_video.mp4")
        assert metadata.file_path == Path("test_video.mp4")
        assert metadata._metadata is None
        assert metadata.errors == []
    
    def test_run_success(self):
        """Test successful metadata extraction."""
        mock_output = {
            "streams": [
                {"codec_type": "video", "width": 1920, "height": 1080},
                {"codec_type": "audio", "codec_name": "aac"}
            ],
            "format": {"duration": "10.5"}
        }
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = json.dumps(mock_output)
            mock_run.return_value.returncode = 0
            
            metadata = Metadata("test_video.mp4")
            result = metadata.run()
            
            assert result == mock_output
            assert metadata._metadata == mock_output
    
    def test_run_ffprobe_error(self):
        """Test ffprobe failure handling."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "ffprobe", "Error")
            
            metadata = Metadata("test_video.mp4")
            with pytest.raises(MetadataError, match="ffprobe failed"):
                metadata.run()
    
    def test_run_json_error(self):
        """Test JSON parsing error handling."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = "invalid json"
            mock_run.return_value.returncode = 0
            
            metadata = Metadata("test_video.mp4")
            with pytest.raises(MetadataError, match="Failed to parse ffprobe output"):
                metadata.run()
    
    def test_metadata_property_not_run(self):
        """Test metadata property when run() hasn't been called."""
        metadata = Metadata("test_video.mp4")
        with pytest.raises(MetadataError, match="FFProbe has not been run yet"):
            _ = metadata.metadata
    
    def test_filter_missing_video(self):
        """Test video stream detection."""
        metadata = Metadata("test_video.mp4")
        metadata._metadata = {
            "streams": [
                {"codec_type": "video", "width": 1920, "height": 1080},
                {"codec_type": "audio", "codec_name": "aac"}
            ]
        }
        
        assert metadata.filter_missing_video() is True
        assert len(metadata.errors) == 0
        
        # Test missing video
        metadata._metadata = {"streams": [{"codec_type": "audio"}]}
        assert metadata.filter_missing_video() is False
        assert len(metadata.errors) == 1
        assert metadata.errors[0]["check"] == "missing_video"
    
    def test_filter_resolution(self):
        """Test resolution filtering."""
        metadata = Metadata("test_video.mp4")
        metadata._metadata = {
            "streams": [{"codec_type": "video", "width": 1920, "height": 1080}]
        }
        
        assert metadata.filter_resolution() is True
        
        # Test low resolution
        metadata._metadata = {
            "streams": [{"codec_type": "video", "width": 100, "height": 100}]
        }
        assert metadata.filter_resolution() is False
        assert len(metadata.errors) == 1
        assert metadata.errors[0]["check"] == "resolution"
    
    def test_filter_duration(self):
        """Test duration filtering."""
        metadata = Metadata("test_video.mp4")
        metadata._metadata = {"format": {"duration": "10.5"}}
        
        assert metadata.filter_duration() is True
        
        # Test short duration
        metadata._metadata = {"format": {"duration": "0.5"}}
        assert metadata.filter_duration() is False
        assert len(metadata.errors) == 1
        assert metadata.errors[0]["check"] == "duration"
    
    def test_export_errors(self):
        """Test error export functionality."""
        metadata = Metadata("test_video.mp4")
        metadata.errors = [{"file": "test.mp4", "check": "test", "message": "test error"}]
        
        json_output = metadata.export_errors()
        assert isinstance(json_output, str)
        
        list_output = metadata.export_errors(as_json=False)
        assert isinstance(list_output, list)
        assert len(list_output) == 1
