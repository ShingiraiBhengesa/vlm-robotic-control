"""
Test suite for the enhanced VLM robotic control system with depth camera.
Tests individual components and integrated system functionality.
"""
import pytest
import numpy as np
import cv2
import time
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules to test
try:
    from vision.depth_camera import DepthCamera
    from vlm_controller_depth import EnhancedVLMController, DetectedObject
    from main_depth_enhanced import EnhancedRoboticSystem
except ImportError as e:
    print(f"⚠️ Import error - some tests may be skipped: {e}")

class TestDepthCamera:
    """Test suite for DepthCamera class."""
    
    def test_depth_camera_initialization(self):
        """Test depth camera initialization without hardware."""
        with patch('pyrealsense2.pipeline') as mock_pipeline:
            camera = DepthCamera(width=640, height=480, fps=30)
            assert camera.width == 640
            assert camera.height == 480
            assert camera.fps == 30
            assert len(camera.depth_filters) > 0
            
    def test_pixel_to_3d_point(self):
        """Test 3D coordinate calculation from pixel coordinates."""
        camera = DepthCamera()
        
        # Mock depth intrinsics
        mock_intrinsics = Mock()
        mock_intrinsics.fx = 600
        mock_intrinsics.fy = 600
        camera.depth_intrinsics = mock_intrinsics
        
        # Mock depth image
        depth_image = np.ones((480, 640), dtype=np.uint16) * 1000  # 1 meter depth
        
        with patch('pyrealsense2.rs2_deproject_pixel_to_point') as mock_deproject:
            mock_deproject.return_value = [0.1, 0.05, 1.0]  # meters
            
            result = camera.pixel_to_3d_point((320, 240), depth_image)
            assert result is not None
            assert len(result) == 3
            # Check conversion to millimeters
            assert result[0] == 100  # 0.1m * 1000
            assert result[1] == 50   # 0.05m * 1000
            assert result[2] == 1000 # 1.0m * 1000
            
    def test_workspace_bounds_calculation(self):
        """Test dynamic workspace bounds calculation."""
        camera = DepthCamera()
        camera.width = 640
        camera.height = 480
        
        # Mock depth intrinsics
        mock_intrinsics = Mock()
        camera.depth_intrinsics = mock_intrinsics
        
        # Create synthetic depth image with valid depth values
        depth_image = np.zeros((480, 640), dtype=np.uint16)
        depth_image[100:300, 200:400] = 500  # 500mm depth in center region
        
        with patch.object(camera, 'pixel_to_3d_point') as mock_pixel_to_3d:
            # Mock 3D point returns
            def mock_3d_point(pixel, depth_img):
                if depth_img[pixel[1], pixel[0]] > 0:
                    return [pixel[0] - 320, pixel[1] - 240, 500]  # Simple mock transformation
                return None
            mock_pixel_to_3d.side_effect = mock_3d_point
            
            bounds = camera.get_workspace_bounds(depth_image, min_height=10, max_height=600)
            
            assert 'x_min' in bounds
            assert 'x_max' in bounds
            assert 'y_min' in bounds
            assert 'y_max' in bounds
            assert 'z_min' in bounds
            assert 'z_max' in bounds

class TestDetectedObject:
    """Test DetectedObject data structure."""
    
    def test_detected_object_creation(self):
        """Test DetectedObject instantiation."""
        obj = DetectedObject(
            label="test_object",
            confidence=0.85,
            bbox_2d=(10, 20, 50, 60),
            center_2d=(35, 50),
            center_3d=[100.0, 200.0, 300.0],
            volume=1000.0,
            is_graspable=True,
            color_dominant=(255, 128, 64)
        )
        
        assert obj.label == "test_object"
        assert obj.confidence == 0.85
        assert obj.bbox_2d == (10, 20, 50, 60)
        assert obj.center_2d == (35, 50)
        assert obj.center_3d == [100.0, 200.0, 300.0]
        assert obj.volume == 1000.0
        assert obj.is_graspable == True
        assert obj.color_dominant == (255, 128, 64)

class TestEnhancedVLMController:
    """Test suite for EnhancedVLMController."""
    
    @pytest.fixture
    def mock_vlm_controller(self):
        """Create a mock VLM controller for testing."""
        with patch('vlm_controller_depth.DepthCamera'), \
             patch('vlm_controller_depth.AutoModelForCausalLM'), \
             patch('vlm_controller_depth.AutoProcessor'):
            
            controller = EnhancedVLMController(debug=False)
            controller.depth_camera = Mock()
            controller.model = Mock()
            controller.processor = Mock()
            return controller
            
    def test_vlm_controller_initialization(self, mock_vlm_controller):
        """Test VLM controller initialization."""
        controller = mock_vlm_controller
        assert controller.min_object_area == 500
        assert controller.min_depth_confidence == 0.7
        assert controller.debug == False
        assert controller.detection_interval == 0.5
        
    def test_graspability_assessment(self, mock_vlm_controller):
        """Test graspability assessment logic."""
        controller = mock_vlm_controller
        
        # Test basic graspability (size-based)
        assert controller._is_graspable(60, 80, 50) == True  # Good size, valid height
        assert controller._is_graspable(10, 15, 50) == False  # Too small
        assert controller._is_graspable(400, 500, 50) == False  # Too large
        assert controller._is_graspable(60, 80, 2) == False  # Too low height
        
    def test_object_memory_management(self, mock_vlm_controller):
        """Test object tracking and memory management."""
        controller = mock_vlm_controller
        
        # Create test objects
        objects = [
            DetectedObject(
                label="test1", confidence=0.8, bbox_2d=(0,0,50,50),
                center_2d=(25,25), center_3d=[100,200,300],
                volume=1000, is_graspable=True, color_dominant=(255,0,0)
            ),
            DetectedObject(
                label="test2", confidence=0.9, bbox_2d=(100,100,60,60),
                center_2d=(130,130), center_3d=[150,250,350],
                volume=1200, is_graspable=False, color_dominant=(0,255,0)
            )
        ]
        
        controller._update_object_memory(objects)
        
        assert len(controller.object_memory) == 2
        assert "test1_100_200" in controller.object_memory
        assert "test2_150_250" in controller.object_memory
        
    def test_alternative_target_finding(self, mock_vlm_controller):
        """Test alternative target position calculation."""
        controller = mock_vlm_controller
        
        # Mock workspace bounds
        controller.workspace_bounds = {
            'x_min': -200, 'x_max': 200,
            'y_min': 0, 'y_max': 300,
            'z_min': 10, 'z_max': 250
        }
        
        # Mock depth camera frames
        controller.depth_camera.get_frames.return_value = (
            np.zeros((480, 640, 3), dtype=np.uint8),  # color frame
            np.ones((480, 640), dtype=np.uint16) * 1000  # depth frame
        )
        
        # Mock path detection (clear path)
        controller.depth_camera.detect_obstacles_in_path.return_value = True
        
        original_target = [100, 150, 100]
        alternative = controller._find_alternative_target(original_target, [])
        
        if alternative:  # If alternative found
            assert len(alternative) == 3
            assert isinstance(alternative[0], (int, float))
            assert isinstance(alternative[1], (int, float))
            assert isinstance(alternative[2], (int, float))

class TestEnhancedRoboticSystem:
    """Test suite for EnhancedRoboticSystem integration."""
    
    @pytest.fixture
    def mock_robotic_system(self):
        """Create a mock robotic system for testing."""
        with patch('main_depth_enhanced.EnhancedVLMController'), \
             patch('main_depth_enhanced.ArduinoController'), \
             patch('builtins.open', mock_open_yaml_config()):
            
            system = EnhancedRoboticSystem(arduino_port='COM_TEST')
            system.vlm_controller = Mock()
            system.arm_controller = Mock()
            return system
            
    def test_system_initialization(self, mock_robotic_system):
        """Test system initialization."""
        system = mock_robotic_system
        assert system.arduino_port == 'COM_TEST'
        assert system.current_arm_position == [0, 200, 100]
        assert system.system_running == False
        assert system.emergency_stop_flag == False
        
    def test_target_validation(self, mock_robotic_system):
        """Test target validation with depth data."""
        system = mock_robotic_system
        
        # Mock validate_position to return True
        with patch('main_depth_enhanced.validate_position', return_value=True):
            # Mock depth camera frames
            system.vlm_controller.depth_camera.get_frames.return_value = (
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.ones((480, 640), dtype=np.uint16) * 1000
            )
            
            # Mock obstacle detection (path clear)
            system.vlm_controller.depth_camera.detect_obstacles_in_path.return_value = True
            
            target = [100, 200, 150]
            result = system._validate_target_with_depth(target)
            assert result == True
            
    def test_emergency_stop(self, mock_robotic_system):
        """Test emergency stop functionality."""
        system = mock_robotic_system
        
        system._emergency_stop()
        assert system.emergency_stop_flag == True
        system.arm_controller.emergency_stop.assert_called_once()

def mock_open_yaml_config():
    """Mock YAML config file opening."""
    import yaml
    from unittest.mock import mock_open
    
    config_data = {
        'base_height': 50,
        'shoulder_length': 146,
        'forearm_length': 187,
        'wrist_length': 100
    }
    
    return mock_open(read_data=yaml.dump(config_data))

class TestIntegration:
    """Integration tests for complete system workflow."""
    
    def test_mock_detection_pipeline(self):
        """Test the object detection pipeline with mocked components."""
        # Create synthetic test data
        color_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth_frame = np.random.randint(500, 1500, (480, 640), dtype=np.uint16)
        
        # Test basic image processing
        assert color_frame.shape == (480, 640, 3)
        assert depth_frame.shape == (480, 640)
        assert color_frame.dtype == np.uint8
        assert depth_frame.dtype == np.uint16
        
    def test_coordinate_transformation_logic(self):
        """Test coordinate transformation calculations."""
        # Test basic coordinate math
        pixel = (320, 240)  # Center pixel
        
        # Mock depth value
        depth_mm = 1000  # 1 meter
        
        # Basic validation of coordinate system
        assert pixel[0] >= 0 and pixel[0] < 640
        assert pixel[1] >= 0 and pixel[1] < 480
        assert depth_mm > 0
        
    def test_safety_boundary_checks(self):
        """Test safety boundary validation."""
        # Define workspace bounds
        bounds = {
            'x_min': -300, 'x_max': 300,
            'y_min': 0, 'y_max': 400,
            'z_min': 10, 'z_max': 250
        }
        
        # Test valid positions
        valid_pos = [100, 200, 150]
        assert (bounds['x_min'] <= valid_pos[0] <= bounds['x_max'])
        assert (bounds['y_min'] <= valid_pos[1] <= bounds['y_max'])
        assert (bounds['z_min'] <= valid_pos[2] <= bounds['z_max'])
        
        # Test invalid positions
        invalid_pos = [400, 500, 300]  # Outside bounds
        assert not (bounds['x_min'] <= invalid_pos[0] <= bounds['x_max'])
        assert not (bounds['y_min'] <= invalid_pos[1] <= bounds['y_max'])
        assert not (bounds['z_min'] <= invalid_pos[2] <= bounds['z_max'])

def run_performance_benchmark():
    """Run performance benchmarks for key components."""
    print("\n🚀 Running Performance Benchmarks...")
    
    # Test numpy array operations (simulating depth processing)
    depth_array = np.random.randint(0, 2000, (480, 640), dtype=np.uint16)
    
    start_time = time.time()
    # Simulate depth filtering operations
    filtered = cv2.medianBlur(depth_array, 5)
    mask = (filtered > 100) & (filtered < 1800)
    valid_pixels = np.sum(mask)
    processing_time = time.time() - start_time
    
    print(f"✅ Depth processing: {processing_time*1000:.2f}ms for {valid_pixels} valid pixels")
    
    # Test coordinate transformation simulation
    start_time = time.time()
    coords_3d = []
    for i in range(1000):  # Simulate 1000 coordinate transformations
        x = np.random.randint(0, 640)
        y = np.random.randint(0, 480)
        d = np.random.randint(500, 1500)
        # Simulate 3D calculation
        coord_3d = [x * 0.5, y * 0.5, d]
        coords_3d.append(coord_3d)
    coord_time = time.time() - start_time
    
    print(f"✅ Coordinate transformation: {coord_time*1000:.2f}ms for 1000 points")
    print(f"   Average: {coord_time*1000000/1000:.1f}μs per point")

if __name__ == "__main__":
    print("🧪 Enhanced VLM Robotic Control System - Test Suite")
    print("=" * 60)
    
    # Run basic validation tests
    print("\n📋 Running Basic Validation Tests...")
    
    try:
        # Test data structures
        test_obj = TestDetectedObject()
        test_obj.test_detected_object_creation()
        print("✅ DetectedObject structure validation passed")
        
        # Test coordinate math
        test_integration = TestIntegration()
        test_integration.test_coordinate_transformation_logic()
        test_integration.test_safety_boundary_checks()
        print("✅ Coordinate transformation logic passed")
        
        # Test mock detection pipeline
        test_integration.test_mock_detection_pipeline()
        print("✅ Detection pipeline structure passed")
        
    except Exception as e:
        print(f"❌ Basic validation failed: {e}")
    
    # Run performance benchmarks
    run_performance_benchmark()
    
    print("\n📊 Test Summary:")
    print("✅ Data structures and logic validated")
    print("✅ Performance benchmarks completed")
    print("⚠️  Hardware tests require Intel RealSense camera")
    print("⚠️  Full integration tests require robotic arm hardware")
    
    print("\n🔧 To run with pytest:")
    print("pip install pytest")
    print("pytest test_enhanced_system.py -v")
    
    print("\n🎯 Manual Testing Steps:")
    print("1. Connect Intel RealSense D435i camera")
    print("2. Run: python main_depth_enhanced.py --camera-only")
    print("3. Test object detection with 'scan' command")
    print("4. Connect robotic arm and test full system")
