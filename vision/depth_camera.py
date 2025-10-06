"""
Stereolabs ZED-M depth camera interface for robotic control.
Provides RGB+Depth streams with direct 3D coordinate mapping.
"""
try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ZED SDK Python bindings not available: {e}")
    print("   Falling back to OpenCV camera interface")
    ZED_AVAILABLE = False
    
import numpy as np
import cv2
import time
from typing import Tuple, Optional, List, Dict
import threading
import queue

class DepthCamera:
    """Advanced depth camera interface using Stereolabs ZED-M with fallback support."""
    
    def __init__(self, 
                 width: int = 640, 
                 height: int = 480, 
                 fps: int = 30,
                 enable_rgb: bool = True,
                 enable_depth: bool = True,
                 depth_mode=None):
        """
        Initialize the ZED-M depth camera with fallback support.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels  
            fps: Frames per second
            enable_rgb: Enable RGB stream
            enable_depth: Enable depth stream
            depth_mode: ZED depth sensing mode (ignored in fallback)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.using_fallback = not ZED_AVAILABLE
        
        # Background thread for continuous capture
        self.capture_thread = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.running = False
        self.camera_info = None
        
        if ZED_AVAILABLE:
            try:
                # Create ZED camera object
                self.zed = sl.Camera()
                
                # Create initialization parameters
                self.init_params = sl.InitParameters()
                self.init_params.camera_resolution = sl.RESOLUTION.VGA  # 640x480
                self.init_params.camera_fps = fps
                self.init_params.depth_mode = depth_mode or sl.DEPTH_MODE.PERFORMANCE
                self.init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use mm for consistency
                self.init_params.depth_minimum_distance = 100  # 10cm minimum
                self.init_params.depth_maximum_distance = 2000  # 2m maximum
                
                # Runtime parameters
                self.runtime_parameters = sl.RuntimeParameters()
                self.runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL  # Fill holes in depth map
                
                # Image containers
                self.image_zed = sl.Mat()
                self.depth_zed = sl.Mat()
                self.point_cloud_zed = sl.Mat()
                
                print(f"✅ ZED-M camera initialized for {width}x{height}@{fps}fps")
            except Exception as e:
                print(f"⚠️ ZED initialization failed: {e}")
                print("   Falling back to OpenCV camera")
                self.using_fallback = True
                
        if self.using_fallback:
            # Fallback to OpenCV camera
            self.cap = cv2.VideoCapture(0)  # Try first camera
            if not self.cap.isOpened():
                # Try other camera indices
                for i in range(1, 5):
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        break
                        
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS, fps)
                print(f"✅ Fallback OpenCV camera initialized for {width}x{height}@{fps}fps")
                print("   Note: Depth functionality will be limited")
            else:
                print("❌ No camera found")
        
    def start(self) -> bool:
        """
        Start the camera (ZED or fallback).
        
        Returns:
            bool: True if started successfully
        """
        if self.using_fallback:
            return self._start_fallback()
        else:
            return self._start_zed()
            
    def _start_zed(self) -> bool:
        """Start ZED camera."""
        try:
            # Open the camera
            err = self.zed.open(self.init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                print(f"❌ Failed to open ZED camera: {err}")
                print("   Switching to fallback mode")
                self.using_fallback = True
                return self._start_fallback()
            
            # Get camera information
            self.camera_info = self.zed.get_camera_information()
            
            # Print camera info
            print(f"✅ ZED-M camera opened successfully")
            print(f"   Serial Number: {self.camera_info.serial_number}")
            print(f"   Firmware: {self.camera_info.camera_firmware_version}")
            print(f"   Resolution: {self.camera_info.camera_resolution}")
            print(f"   FPS: {self.zed.get_init_parameters().camera_fps}")
            
            # Start background capture thread
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop_zed)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start ZED camera: {e}")
            print("   Switching to fallback mode")
            self.using_fallback = True
            return self._start_fallback()
            
    def _start_fallback(self) -> bool:
        """Start fallback OpenCV camera."""
        try:
            if not hasattr(self, 'cap') or not self.cap.isOpened():
                print("❌ Fallback camera not available")
                return False
                
            print("✅ Fallback camera started")
            print("   Note: Using RGB-only mode (no depth data)")
            
            # Start background capture thread
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop_fallback)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start fallback camera: {e}")
            return False
            
    def _capture_loop_zed(self):
        """Background thread for ZED camera frame capture."""
        while self.running:
            try:
                # Grab an image
                if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    # Retrieve left image (RGB)
                    self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT)
                    
                    # Retrieve depth map
                    self.zed.retrieve_measure(self.depth_zed, sl.MEASURE.DEPTH)
                    
                    # Convert to numpy arrays
                    image_np = self.image_zed.get_data()
                    depth_np = self.depth_zed.get_data()
                    
                    # Add to queue (drop old frames if queue is full)
                    frame_data = {
                        'color_frame': image_np,
                        'depth_frame': depth_np,
                        'timestamp': time.time()
                    }
                    
                    try:
                        self.frame_queue.put_nowait(frame_data)
                    except queue.Full:
                        # Remove oldest frame and add new one
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame_data)
                        except queue.Empty:
                            pass
                else:
                    time.sleep(0.001)  # Small delay if grab failed
                        
            except Exception as e:
                if self.running:
                    print(f"⚠️ Frame capture error: {e}")
                time.sleep(0.01)
                
    def _capture_loop_fallback(self):
        """Background thread for fallback OpenCV camera capture."""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    # Create mock depth frame (all zeros)
                    depth_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
                    
                    # Add to queue (drop old frames if queue is full)
                    frame_data = {
                        'color_frame': frame,
                        'depth_frame': depth_frame,
                        'timestamp': time.time()
                    }
                    
                    try:
                        self.frame_queue.put_nowait(frame_data)
                    except queue.Full:
                        # Remove oldest frame and add new one
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame_data)
                        except queue.Empty:
                            pass
                else:
                    time.sleep(0.001)  # Small delay if grab failed
                        
            except Exception as e:
                if self.running:
                    print(f"⚠️ Fallback frame capture error: {e}")
                time.sleep(0.01)
                
    def get_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get the latest RGB and depth frames.
        
        Returns:
            tuple: (color_image, depth_image) or (None, None) if failed
        """
        try:
            frame_data = self.frame_queue.get_nowait()
            
            color_frame = frame_data['color_frame']
            depth_frame = frame_data['depth_frame']
            
            # Convert RGBA to BGR for OpenCV compatibility
            if color_frame is not None and color_frame.shape[2] == 4:
                color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGBA2BGR)
            
            return color_frame, depth_frame
            
        except queue.Empty:
            return None, None
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
            return None, None
            
    def pixel_to_3d_point(self, pixel: Tuple[int, int], depth_image: np.ndarray) -> Optional[List[float]]:
        """
        Convert 2D pixel coordinates to 3D world coordinates using ZED depth data.
        
        Args:
            pixel: (x, y) pixel coordinates
            depth_image: Depth image from ZED camera
            
        Returns:
            list: [x, y, z] coordinates in millimeters, or None if invalid
        """
        if not self.camera_info:
            print("❌ Camera not properly initialized")
            return None
            
        try:
            x, y = pixel
            if not (0 <= x < self.width and 0 <= y < self.height):
                return None
                
            # Get depth value at pixel (in millimeters from ZED)
            depth_value = depth_image[y, x]
            
            # Check for invalid depth
            if not np.isfinite(depth_value) or depth_value <= 0:
                return None
                
            # Get camera calibration parameters
            calibration_params = self.camera_info.calibration_parameters.left_cam
            fx = calibration_params.fx
            fy = calibration_params.fy
            cx = calibration_params.cx
            cy = calibration_params.cy
            
            # Convert pixel to 3D point using pinhole camera model
            z = depth_value  # Depth in mm
            x_3d = (x - cx) * z / fx
            y_3d = (y - cy) * z / fy
            
            return [float(x_3d), float(y_3d), float(z)]
            
        except Exception as e:
            print(f"❌ 3D projection error: {e}")
            return None
            
    def get_point_cloud(self) -> Optional[np.ndarray]:
        """
        Generate point cloud from current ZED frame.
        
        Returns:
            np.ndarray: Point cloud array (N, 4) with [x, y, z, color] or None if failed
        """
        try:
            if not self.running or not self.camera_info:
                return None
                
            # Retrieve point cloud
            self.zed.retrieve_measure(self.point_cloud_zed, sl.MEASURE.XYZRGBA)
            point_cloud_data = self.point_cloud_zed.get_data()
            
            if point_cloud_data is not None:
                # Reshape point cloud data
                pc_reshaped = point_cloud_data.reshape(-1, 4)
                
                # Filter out invalid points (NaN or infinite)
                valid_mask = np.isfinite(pc_reshaped[:, :3]).all(axis=1)
                valid_points = pc_reshaped[valid_mask]
                
                return valid_points
            
            return None
            
        except Exception as e:
            print(f"❌ Point cloud generation error: {e}")
            return None
            
    def detect_obstacles_in_path(self, start_3d: List[float], end_3d: List[float], 
                                depth_image: np.ndarray, safety_margin: float = 50.0) -> bool:
        """
        Check if there are obstacles along a 3D path.
        
        Args:
            start_3d: Starting 3D coordinate [x, y, z] in mm
            end_3d: Ending 3D coordinate [x, y, z] in mm  
            depth_image: Current depth image from ZED
            safety_margin: Safety margin in mm
            
        Returns:
            bool: True if path is clear, False if obstacles detected
        """
        try:
            # Sample points along the path
            num_samples = 20
            for i in range(num_samples):
                t = i / (num_samples - 1)
                sample_point = [
                    start_3d[0] + t * (end_3d[0] - start_3d[0]),
                    start_3d[1] + t * (end_3d[1] - start_3d[1]), 
                    start_3d[2] + t * (end_3d[2] - start_3d[2])
                ]
                
                # Project 3D point back to pixel
                pixel = self._project_3d_to_pixel(sample_point)
                if not pixel:
                    continue
                    
                x, y = pixel
                if not (0 <= x < self.width and 0 <= y < self.height):
                    continue
                    
                # Check surrounding area for obstacles
                window_size = 5
                for dx in range(-window_size, window_size + 1):
                    for dy in range(-window_size, window_size + 1):
                        px, py = x + dx, y + dy
                        if not (0 <= px < self.width and 0 <= py < self.height):
                            continue
                            
                        depth_value = depth_image[py, px]
                        if not np.isfinite(depth_value) or depth_value <= 0:
                            continue
                            
                        # Convert to 3D and check if too close to path
                        obstacle_3d = self.pixel_to_3d_point((px, py), depth_image)
                        if not obstacle_3d:
                            continue
                            
                        # Calculate distance from obstacle to path point
                        distance = np.linalg.norm([
                            obstacle_3d[0] - sample_point[0],
                            obstacle_3d[1] - sample_point[1], 
                            obstacle_3d[2] - sample_point[2]
                        ])
                        
                        if distance < safety_margin:
                            print(f"⚠️ Obstacle detected at {obstacle_3d} (distance: {distance:.1f}mm)")
                            return False
                            
            return True
            
        except Exception as e:
            print(f"❌ Path obstacle detection error: {e}")
            return False
            
    def _project_3d_to_pixel(self, point_3d: List[float]) -> Optional[Tuple[int, int]]:
        """Project 3D point back to 2D pixel coordinates using ZED calibration."""
        try:
            if not self.camera_info:
                return None
                
            # Get camera calibration parameters
            calibration_params = self.camera_info.calibration_parameters.left_cam
            fx = calibration_params.fx
            fy = calibration_params.fy
            cx = calibration_params.cx
            cy = calibration_params.cy
            
            x_3d, y_3d, z_3d = point_3d
            
            if z_3d <= 0:
                return None
                
            # Project using pinhole camera model
            u = int((x_3d * fx / z_3d) + cx)
            v = int((y_3d * fy / z_3d) + cy)
            
            return (u, v)
            
        except Exception:
            return None
            
    def get_workspace_bounds(self, depth_image: np.ndarray, 
                           min_height: float = 10.0, max_height: float = 2000.0) -> Dict:
        """
        Dynamically determine workspace bounds from ZED depth data.
        
        Args:
            depth_image: Current depth image from ZED
            min_height: Minimum Z coordinate in mm
            max_height: Maximum Z coordinate in mm
            
        Returns:
            dict: Workspace bounds {x_min, x_max, y_min, y_max, z_min, z_max}
        """
        try:
            valid_points = []
            
            # Sample points across the image
            step = 10
            for y in range(0, self.height, step):
                for x in range(0, self.width, step):
                    depth_value = depth_image[y, x]
                    if not np.isfinite(depth_value) or depth_value <= 0:
                        continue
                        
                    point_3d = self.pixel_to_3d_point((x, y), depth_image)
                    if not point_3d:
                        continue
                        
                    # Filter by height bounds
                    if min_height <= point_3d[2] <= max_height:
                        valid_points.append(point_3d)
                        
            if not valid_points:
                # Return default bounds if no valid points
                return {
                    'x_min': -300, 'x_max': 300,
                    'y_min': 0, 'y_max': 400,
                    'z_min': 100, 'z_max': 2000
                }
                
            points_array = np.array(valid_points)
            
            return {
                'x_min': float(np.min(points_array[:, 0])),
                'x_max': float(np.max(points_array[:, 0])),
                'y_min': float(np.min(points_array[:, 1])),
                'y_max': float(np.max(points_array[:, 1])),
                'z_min': float(np.max([min_height, np.min(points_array[:, 2])])),
                'z_max': float(np.min([max_height, np.max(points_array[:, 2])]))
            }
            
        except Exception as e:
            print(f"❌ Workspace bounds calculation error: {e}")
            return {
                'x_min': -300, 'x_max': 300,
                'y_min': 0, 'y_max': 400, 
                'z_min': 100, 'z_max': 2000
            }
            
    def get_camera_info(self) -> Dict:
        """Get ZED camera information and calibration parameters."""
        if not self.camera_info:
            return {}
            
        try:
            left_cam = self.camera_info.calibration_parameters.left_cam
            return {
                'serial_number': self.camera_info.serial_number,
                'firmware_version': self.camera_info.camera_firmware_version,
                'resolution': str(self.camera_info.camera_resolution),
                'fx': left_cam.fx,
                'fy': left_cam.fy,
                'cx': left_cam.cx,
                'cy': left_cam.cy,
                'baseline': self.camera_info.calibration_parameters.T[0],  # Stereo baseline
                'distortion': left_cam.disto
            }
        except Exception as e:
            print(f"⚠️ Error getting camera info: {e}")
            return {}
    
    def set_camera_settings(self, brightness: int = 4, contrast: int = 4, 
                           hue: int = 0, saturation: int = 4, sharpness: int = 4):
        """
        Adjust ZED camera settings for optimal performance.
        
        Args:
            brightness: Brightness level (0-8)
            contrast: Contrast level (0-8) 
            hue: Hue level (0-11)
            saturation: Saturation level (0-8)
            sharpness: Sharpness level (0-8)
        """
        try:
            if self.zed.is_opened():
                self.zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, brightness)
                self.zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, contrast)
                self.zed.set_camera_settings(sl.VIDEO_SETTINGS.HUE, hue)
                self.zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, saturation)
                self.zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, sharpness)
                print(f"✅ ZED camera settings updated")
            
        except Exception as e:
            print(f"⚠️ Failed to set camera settings: {e}")
            
    def stop(self):
        """Stop the ZED camera and cleanup."""
        try:
            self.running = False
            if self.capture_thread:
                self.capture_thread.join(timeout=2.0)
                
            if self.zed.is_opened():
                self.zed.close()
            print("✅ ZED-M camera stopped")
            
        except Exception as e:
            print(f"⚠️ Error stopping ZED camera: {e}")
            
    def __del__(self):
        """Destructor - ensure camera is properly closed."""
        self.stop()
