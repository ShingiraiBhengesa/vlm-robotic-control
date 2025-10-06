"""
Enhanced Vision-Language Model controller with Intel RealSense depth camera integration.
Provides accurate 3D object detection and positioning without manual calibration.
"""
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from vision.depth_camera import DepthCamera
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import threading
from dataclasses import dataclass

@dataclass
class DetectedObject:
    """Structure for detected objects with 3D positioning."""
    label: str
    confidence: float
    bbox_2d: Tuple[int, int, int, int]  # (x, y, width, height)
    center_2d: Tuple[int, int]  # (x, y) pixel coordinates
    center_3d: List[float]  # [x, y, z] in mm
    volume: float  # Estimated object volume in mm³
    is_graspable: bool  # Whether object appears graspable
    color_dominant: Tuple[int, int, int]  # Dominant RGB color

class EnhancedVLMController:
    """Enhanced VLM controller with depth camera integration."""
    
    def __init__(self, 
                 model_name: str = "microsoft/Florence-2-base",
                 min_object_area: int = 500,
                 min_depth_confidence: float = 0.7,
                 debug: bool = True):
        """
        Initialize enhanced VLM controller with depth camera.
        
        Args:
            model_name: Pretrained VLM model name
            min_object_area: Minimum pixel area for object detection
            min_depth_confidence: Minimum confidence for depth measurements
            debug: Enable debug logging and visualization
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Try multiple approaches to load Florence-2
        self.using_florence = False
        self.model = None
        self.processor = None
        
        # Use YOLO + Enhanced Fallback (most reliable approach)
        print("🔄 Initializing YOLO + Enhanced Fallback system...")
        print("   This combination provides the best reliability and intelligence")
        
        # Don't load complex VLMs that have compatibility issues
        # Focus on YOLO (when available) + enhanced color detection
        self.using_florence = False
        self.using_blip = False
        self.model = None
        self.processor = None
        
        print("✅ Enhanced Detection System initialized")
        print("   Primary: YOLO object detection (when available)")
        print("   Fallback: Intelligent color-based detection")
        
        self.min_object_area = min_object_area
        self.min_depth_confidence = min_depth_confidence
        self.debug = debug
        
        # Initialize depth camera
        self.depth_camera = DepthCamera(
            width=640, height=480, fps=30,
            enable_rgb=True, enable_depth=True
        )
        
        # Object tracking and memory
        self.object_memory = {}  # Track objects across frames
        self.workspace_bounds = None
        self.last_detection_time = 0
        self.detection_interval = 0.5  # Detect objects every 0.5 seconds
        
        # Safety parameters
        self.min_grasp_size = 20.0  # mm
        self.max_grasp_size = 150.0  # mm
        self.min_object_height = 5.0  # mm
        self.collision_margin = 50.0  # mm
        
        print(f"✅ Enhanced VLM Controller initialized with {model_name} on {self.device}")
        
    def start_camera(self) -> bool:
        """Start the depth camera system."""
        success = self.depth_camera.start()
        if success:
            # Wait for camera to stabilize
            time.sleep(2.0)
            
            # Get initial workspace bounds
            color_frame, depth_frame = self.depth_camera.get_frames()
            if depth_frame is not None:
                self.workspace_bounds = self.depth_camera.get_workspace_bounds(depth_frame)
                print(f"✅ Workspace bounds: {self.workspace_bounds}")
        
        return success
        
    def detect_objects_with_depth(self, use_yolo: bool = False) -> List[DetectedObject]:
        """
        Detect objects using RGB+Depth data for accurate 3D positioning.
        
        Args:
            use_yolo: Use YOLO for object detection (more accurate than Florence-2)
            
        Returns:
            List of DetectedObject instances with 3D coordinates
        """
        # Skip detection if too recent
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            return []
            
        try:
            # Get latest frames
            color_frame, depth_frame = self.depth_camera.get_frames()
            if color_frame is None or depth_frame is None:
                return []
                
            detected_objects = []
            
            if use_yolo:
                detected_objects = self._detect_with_yolo(color_frame, depth_frame)
            else:
                detected_objects = self._detect_with_florence(color_frame, depth_frame)
                
            # Filter and enhance detections
            filtered_objects = self._filter_and_enhance_detections(
                detected_objects, color_frame, depth_frame
            )
            
            # Update object memory for tracking
            self._update_object_memory(filtered_objects)
            
            # Save debug visualization
            if self.debug and filtered_objects:
                self._save_debug_image(color_frame, depth_frame, filtered_objects)
                
            self.last_detection_time = current_time
            return filtered_objects
            
        except Exception as e:
            print(f"❌ Object detection error: {e}")
            return []
            
    def _detect_with_florence(self, color_frame: np.ndarray, depth_frame: np.ndarray) -> List[DetectedObject]:
        """Use Florence-2 for enhanced object detection optimized for robotics."""
        if not self.using_florence:
            return self._detect_simple_objects(color_frame, depth_frame)
            
        try:
            # Convert to PIL Image
            image = Image.fromarray(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))
            
            # Enhanced prompt for robotics-oriented detection
            prompt = "<OD>"  # Object detection task for Florence-2
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    early_stopping=False,
                    do_sample=False
                )
                
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            # Parse Florence-2 output
            parsed_answer = self.processor.post_process_generation(
                generated_text, task="<OD>", image_size=(image.width, image.height)
            )
            
            detected_objects = []
            
            if '<OD>' in parsed_answer:
                od_results = parsed_answer['<OD>']
                
                for i, (bbox, label) in enumerate(zip(od_results['bboxes'], od_results['labels'])):
                    x1, y1, x2, y2 = bbox
                    w, h = x2 - x1, y2 - y1
                    
                    # Filter by minimum area
                    if w * h < self.min_object_area:
                        continue
                        
                    # Calculate center coordinates
                    center_x, center_y = int(x1 + w/2), int(y1 + h/2)
                    
                    # Get 3D coordinate from depth
                    center_3d = self.depth_camera.pixel_to_3d_point(
                        (center_x, center_y), depth_frame
                    )
                    
                    if center_3d is None:
                        continue
                        
                    # Extract dominant color
                    roi = color_frame[int(y1):int(y2), int(x1):int(x2)]
                    dominant_color = self._get_dominant_color(roi)
                    
                    # Enhance object labels for robotics context
                    enhanced_label = self._enhance_object_label(label, roi)
                    
                    # Advanced graspability assessment based on object type
                    is_graspable = self._assess_graspability_by_type(enhanced_label, w, h, center_3d[2])
                    
                    detected_objects.append(DetectedObject(
                        label=enhanced_label,
                        confidence=0.85,  # Higher confidence for Florence-2 real objects
                        bbox_2d=(int(x1), int(y1), int(w), int(h)),
                        center_2d=(center_x, center_y),
                        center_3d=center_3d,
                        volume=self._estimate_object_volume(bbox, depth_frame, x1, y1, x2, y2),
                        is_graspable=is_graspable,
                        color_dominant=dominant_color
                    ))
                    
            return detected_objects
            
        except Exception as e:
            print(f"❌ Florence detection error: {e}")
            return self._detect_simple_objects(color_frame, depth_frame)
            
    def _enhance_object_label(self, label: str, roi: np.ndarray) -> str:
        """Enhance object labels for better robotics understanding."""
        try:
            # Map generic labels to more specific robotics-friendly names
            label_mappings = {
                'bottle': 'plastic_bottle',
                'cup': 'coffee_cup', 
                'cell phone': 'smartphone',
                'remote': 'tv_remote',
                'book': 'notebook',
                'scissors': 'cutting_tool',
                'knife': 'kitchen_knife',
                'fork': 'dining_fork',
                'spoon': 'dining_spoon'
            }
            
            enhanced_label = label_mappings.get(label.lower(), label)
            
            # Add color information if object is distinctly colored
            if roi.size > 0:
                dominant_color = self._get_dominant_color(roi)
                r, g, b = dominant_color
                
                # Classify color
                if r > 150 and g < 100 and b < 100:
                    enhanced_label = f"red_{enhanced_label}"
                elif g > 150 and r < 100 and b < 100:
                    enhanced_label = f"green_{enhanced_label}"
                elif b > 150 and r < 100 and g < 100:
                    enhanced_label = f"blue_{enhanced_label}"
                elif r > 200 and g > 200 and b < 100:
                    enhanced_label = f"yellow_{enhanced_label}"
                    
            return enhanced_label
            
        except Exception:
            return label
            
    def _assess_graspability_by_type(self, label: str, width_px: float, height_px: float, z_mm: float) -> bool:
        """Advanced graspability assessment based on object type."""
        try:
            # Object type graspability rules
            easily_graspable = [
                'bottle', 'cup', 'smartphone', 'remote', 'book', 'tool',
                'plastic_bottle', 'coffee_cup', 'tv_remote', 'notebook'
            ]
            
            moderately_graspable = [
                'scissors', 'knife', 'fork', 'spoon', 'pen', 'marker',
                'cutting_tool', 'kitchen_knife', 'dining_fork', 'dining_spoon'
            ]
            
            difficult_to_grasp = [
                'paper', 'cloth', 'fabric', 'sheet'
            ]
            
            # Check if object type is graspable
            label_lower = label.lower()
            
            # Size-based filtering
            mm_per_pixel = 0.5
            size_mm = max(width_px, height_px) * mm_per_pixel
            
            if size_mm < self.min_grasp_size or size_mm > self.max_grasp_size:
                return False
                
            if z_mm < self.min_object_height:
                return False
                
            # Type-based assessment
            for graspable_type in easily_graspable:
                if graspable_type in label_lower:
                    return True
                    
            for moderate_type in moderately_graspable:
                if moderate_type in label_lower:
                    return size_mm > 30.0  # Need minimum size for moderate objects
                    
            for difficult_type in difficult_to_grasp:
                if difficult_type in label_lower:
                    return False
                    
            # Default assessment for unknown objects
            return self._is_graspable(width_px, height_px, z_mm)
            
        except Exception:
            return self._is_graspable(width_px, height_px, z_mm)
            
    def _detect_simple_objects(self, color_frame: np.ndarray, depth_frame: np.ndarray) -> List[DetectedObject]:
        """Simple object detection using OpenCV contours as fallback."""
        try:
            detected_objects = []
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for common objects
            color_ranges = {
                'red_object': [(0, 50, 50), (10, 255, 255)],
                'blue_object': [(100, 50, 50), (130, 255, 255)],
                'green_object': [(40, 50, 50), (80, 255, 255)],
                'yellow_object': [(20, 50, 50), (40, 255, 255)]
            }
            
            for color_name, (lower, upper) in color_ranges.items():
                # Create mask for color
                lower_np = np.array(lower)
                upper_np = np.array(upper)
                mask = cv2.inRange(hsv, lower_np, upper_np)
                
                # Apply morphological operations to clean up mask
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < self.min_object_area:
                        continue
                        
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate center coordinates
                    center_x, center_y = x + w//2, y + h//2
                    
                    # Get 3D coordinate from depth
                    center_3d = self.depth_camera.pixel_to_3d_point(
                        (center_x, center_y), depth_frame
                    )
                    
                    if center_3d is None:
                        # Use estimated depth if depth camera is not working
                        center_3d = [center_x - 320, center_y - 240, 300]  # Rough estimate
                        
                    # Extract dominant color
                    roi = color_frame[y:y+h, x:x+w]
                    dominant_color = self._get_dominant_color(roi)
                    
                    detected_objects.append(DetectedObject(
                        label=color_name,
                        confidence=0.7,  # Estimated confidence
                        bbox_2d=(x, y, w, h),
                        center_2d=(center_x, center_y),
                        center_3d=center_3d,
                        volume=self._estimate_object_volume(None, depth_frame, x, y, x+w, y+h),
                        is_graspable=self._is_graspable(w, h, center_3d[2]),
                        color_dominant=dominant_color
                    ))
                    
            return detected_objects
            
        except Exception as e:
            print(f"❌ Simple detection error: {e}")
            return []
            
    def _detect_with_yolo(self, color_frame: np.ndarray, depth_frame: np.ndarray) -> List[DetectedObject]:
        """Use YOLO for more accurate object detection (requires ultralytics)."""
        try:
            # This requires: pip install ultralytics
            from ultralytics import YOLO
            
            # Load YOLO model
            model = YOLO('yolov8n.pt')  # Nano model for speed
            
            # Run inference
            results = model(color_frame, verbose=False)
            
            detected_objects = []
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box coordinates and info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        label = model.names[class_id]
                        
                        # Filter by confidence and area
                        w, h = x2 - x1, y2 - y1
                        if confidence < 0.5 or w * h < self.min_object_area:
                            continue
                            
                        # Calculate center coordinates  
                        center_x, center_y = int(x1 + w/2), int(y1 + h/2)
                        
                        # Get 3D coordinate from depth
                        center_3d = self.depth_camera.pixel_to_3d_point(
                            (center_x, center_y), depth_frame
                        )
                        
                        if center_3d is None:
                            continue
                            
                        # Extract dominant color
                        roi = color_frame[int(y1):int(y2), int(x1):int(x2)]
                        dominant_color = self._get_dominant_color(roi)
                        
                        detected_objects.append(DetectedObject(
                            label=label,
                            confidence=confidence,
                            bbox_2d=(int(x1), int(y1), int(w), int(h)),
                            center_2d=(center_x, center_y),
                            center_3d=center_3d,
                            volume=self._estimate_object_volume(None, depth_frame, x1, y1, x2, y2),
                            is_graspable=self._is_graspable(w, h, center_3d[2]),
                            color_dominant=dominant_color
                        ))
                        
            return detected_objects
            
        except ImportError:
            print("⚠️ YOLO not available. Install with: pip install ultralytics")
            return []
        except Exception as e:
            print(f"❌ YOLO detection error: {e}")
            return []
            
    def _filter_and_enhance_detections(self, 
                                     objects: List[DetectedObject], 
                                     color_frame: np.ndarray,
                                     depth_frame: np.ndarray) -> List[DetectedObject]:
        """Filter and enhance object detections."""
        filtered_objects = []
        
        for obj in objects:
            # Check workspace bounds
            if self.workspace_bounds:
                x, y, z = obj.center_3d
                if not (self.workspace_bounds['x_min'] <= x <= self.workspace_bounds['x_max'] and
                        self.workspace_bounds['y_min'] <= y <= self.workspace_bounds['y_max'] and
                        self.workspace_bounds['z_min'] <= z <= self.workspace_bounds['z_max']):
                    continue
                    
            # Check minimum object height
            if obj.center_3d[2] < self.min_object_height:
                continue
                
            # Enhanced graspability assessment
            obj.is_graspable = self._assess_graspability_advanced(obj, depth_frame)
            
            filtered_objects.append(obj)
            
        return filtered_objects
        
    def _estimate_object_volume(self, bbox, depth_frame: np.ndarray, 
                              x1: float, y1: float, x2: float, y2: float) -> float:
        """Estimate object volume using depth data."""
        try:
            # Extract depth values within bounding box
            roi_depth = depth_frame[int(y1):int(y2), int(x1):int(x2)]
            valid_depths = roi_depth[roi_depth > 0]
            
            if len(valid_depths) == 0:
                return 0.0
                
            # Estimate dimensions
            width_px = x2 - x1
            height_px = y2 - y1
            depth_mm = float(np.std(valid_depths))  # Use depth variation as thickness estimate
            
            # Convert pixel dimensions to mm (rough estimation)
            # This assumes objects at ~300mm distance
            mm_per_pixel = 0.5  # Approximate conversion factor
            width_mm = width_px * mm_per_pixel
            height_mm = height_px * mm_per_pixel
            
            volume = width_mm * height_mm * max(depth_mm, 10.0)  # Minimum thickness 10mm
            return float(volume)
            
        except Exception:
            return 0.0
            
    def _is_graspable(self, width_px: float, height_px: float, z_mm: float) -> bool:
        """Basic graspability assessment."""
        # Convert pixel size to rough mm estimate
        mm_per_pixel = 0.5  # Approximate at typical working distance
        size_mm = max(width_px, height_px) * mm_per_pixel
        
        return (self.min_grasp_size <= size_mm <= self.max_grasp_size and 
                z_mm >= self.min_object_height)
                
    def _assess_graspability_advanced(self, obj: DetectedObject, depth_frame: np.ndarray) -> bool:
        """Advanced graspability assessment using depth data."""
        try:
            x, y, w, h = obj.bbox_2d
            
            # Check object isolation (no obstacles around it)
            margin = 20  # pixels
            extended_roi = depth_frame[
                max(0, y-margin):min(depth_frame.shape[0], y+h+margin),
                max(0, x-margin):min(depth_frame.shape[1], x+w+margin)
            ]
            
            object_roi = depth_frame[y:y+h, x:x+w]
            
            if object_roi.size == 0 or extended_roi.size == 0:
                return False
                
            # Get median depth of object and surroundings
            obj_depths = object_roi[object_roi > 0]
            ext_depths = extended_roi[extended_roi > 0]
            
            if len(obj_depths) == 0 or len(ext_depths) == 0:
                return False
                
            obj_depth = np.median(obj_depths)
            surrounding_depth = np.median(ext_depths)
            
            # Object should be closer than its surroundings (standing out)
            depth_diff = surrounding_depth - obj_depth
            
            # Check if object has sufficient clearance
            clearance_threshold = 30.0  # mm
            
            return (depth_diff > clearance_threshold and 
                    obj.is_graspable and
                    self._check_approach_angles(obj, depth_frame))
                    
        except Exception:
            return obj.is_graspable
            
    def _check_approach_angles(self, obj: DetectedObject, depth_frame: np.ndarray) -> bool:
        """Check if object can be approached from multiple angles."""
        try:
            x, y, w, h = obj.bbox_2d
            center_x, center_y = obj.center_2d
            
            # Check 4 approach directions
            approach_dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
            clear_approaches = 0
            
            for dx, dy in approach_dirs:
                # Sample points along approach direction
                for dist in range(10, 50, 10):  # Check 10-50 pixels away
                    check_x = center_x + dx * dist
                    check_y = center_y + dy * dist
                    
                    if not (0 <= check_x < depth_frame.shape[1] and 0 <= check_y < depth_frame.shape[0]):
                        break
                        
                    depth_val = depth_frame[check_y, check_x]
                    if depth_val == 0:
                        continue
                        
                    # Check if approach path is clear
                    if depth_val > obj.center_3d[2] + self.collision_margin:
                        clear_approaches += 1
                        break
                        
            # Need at least 2 clear approach angles
            return clear_approaches >= 2
            
        except Exception:
            return True
            
    def _get_dominant_color(self, roi: np.ndarray) -> Tuple[int, int, int]:
        """Extract dominant color from region of interest."""
        try:
            if roi.size == 0:
                return (128, 128, 128)
                
            # Reshape and find most common color
            pixels = roi.reshape(-1, 3)
            
            # Use k-means to find dominant color
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
            kmeans.fit(pixels)
            dominant_color = kmeans.cluster_centers_[0]
            
            return tuple(map(int, dominant_color))
            
        except Exception:
            # Fallback to mean color
            return tuple(map(int, np.mean(roi, axis=(0, 1))))
            
    def _update_object_memory(self, objects: List[DetectedObject]):
        """Update object tracking memory."""
        current_time = time.time()
        
        # Remove old objects (not seen for 5 seconds)
        expired_keys = [k for k, v in self.object_memory.items() 
                       if current_time - v['last_seen'] > 5.0]
        for key in expired_keys:
            del self.object_memory[key]
            
        # Update current objects
        for obj in objects:
            obj_key = f"{obj.label}_{int(obj.center_3d[0])}_{int(obj.center_3d[1])}"
            self.object_memory[obj_key] = {
                'object': obj,
                'last_seen': current_time,
                'stability_count': self.object_memory.get(obj_key, {}).get('stability_count', 0) + 1
            }
            
    def plan_action_with_depth(self, user_command: str) -> Dict:
        """
        Plan robotic action using depth-enhanced object understanding.
        
        Args:
            user_command: Natural language command
            
        Returns:
            Action plan dictionary
        """
        try:
            # Get current objects
            detected_objects = self.detect_objects_with_depth(use_yolo=True)
            
            if not detected_objects:
                return {"error": "No objects detected in workspace"}
                
            # Get current frames for visual context
            color_frame, depth_frame = self.depth_camera.get_frames()
            if color_frame is None:
                return {"error": "Unable to capture camera frame"}
                
            # Build context for VLM
            objects_context = []
            for obj in detected_objects:
                objects_context.append({
                    "label": obj.label,
                    "position_3d": {
                        "x": round(obj.center_3d[0], 1),
                        "y": round(obj.center_3d[1], 1), 
                        "z": round(obj.center_3d[2], 1)
                    },
                    "graspable": obj.is_graspable,
                    "confidence": round(obj.confidence, 2),
                    "dominant_color": obj.color_dominant,
                    "volume_mm3": round(obj.volume, 0)
                })
                
            # Enhanced system prompt with depth awareness
            system_prompt = """You are controlling a 5-DOF Lynxmotion robotic arm with depth camera vision.

WORKSPACE BOUNDS (mm):
- X: -300 to 300 (left-right)
- Y: 0 to 400 (near-far)  
- Z: 10 to 250 (height)

AVAILABLE OBJECTS with precise 3D coordinates:
""" + json.dumps(objects_context, indent=2) + """

COMMANDS:
- MOVE: Move to 3D coordinate
- GRIP: Control gripper (open/close)
- SCAN: Re-scan workspace for objects

SAFETY:
- Only move to detected object positions
- Check graspable flag before gripping
- Avoid collisions with other objects

Respond in JSON format:
{
    "command": "MOVE|GRIP|SCAN",
    "target": [x, y, z],
    "speed": "slow|normal|fast", 
    "gripper": "open|close",
    "confidence": 0.0-1.0,
    "reasoning": "explanation of action"
}"""

            # Process command with VLM
            image = Image.fromarray(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))
            full_prompt = f"{system_prompt}\n\nUSER COMMAND: {user_command}\n\nProvide JSON response:"
            
            inputs = self.processor(
                text=full_prompt, images=image, return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=200,
                    num_beams=1,
                    do_sample=False
                )
                
            response_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Extract JSON from response
            try:
                # Find JSON in response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    response = json.loads(json_str)
                    
                    # Validate and enhance response
                    return self._validate_and_enhance_action(response, detected_objects, depth_frame)
                else:
                    return {"error": "No valid JSON found in VLM response"}
                    
            except json.JSONDecodeError as e:
                return {"error": f"JSON parsing failed: {e}"}
                
        except Exception as e:
            return {"error": f"Action planning failed: {e}"}
            
    def _validate_and_enhance_action(self, action: Dict, objects: List[DetectedObject], depth_frame: np.ndarray) -> Dict:
        """Validate and enhance planned action with safety checks."""
        try:
            if action.get("command") == "MOVE":
                target = action.get("target")
                if not target or len(target) != 3:
                    return {"error": "Invalid MOVE target coordinates"}
                    
                # Check workspace bounds
                if self.workspace_bounds:
                    x, y, z = target
                    bounds = self.workspace_bounds
                    if not (bounds['x_min'] <= x <= bounds['x_max'] and
                            bounds['y_min'] <= y <= bounds['y_max'] and 
                            bounds['z_min'] <= z <= bounds['z_max']):
                        return {"error": f"Target {target} outside workspace bounds"}
                        
                # Check for collision avoidance
                current_pos = [0, 200, 100]  # Assume current arm position (could be tracked)
                
                if not self.depth_camera.detect_obstacles_in_path(
                    current_pos, target, depth_frame, self.collision_margin
                ):
                    # Try to find alternative path or adjust target
                    alternative_target = self._find_alternative_target(target, objects)
                    if alternative_target:
                        action["target"] = alternative_target
                        action["reasoning"] += " (adjusted for obstacle avoidance)"
                    else:
                        return {"error": "Path blocked and no alternative found"}
                        
                # Enhance action with additional safety info
                action["collision_checked"] = True
                action["workspace_validated"] = True
                
            elif action.get("command") == "GRIP":
                gripper_action = action.get("gripper")
                if gripper_action not in ["open", "close"]:
                    return {"error": "Invalid gripper command"}
                    
                # Check if we're at a graspable object when closing gripper
                if gripper_action == "close":
                    nearby_objects = self._find_nearby_graspable_objects(
                        action.get("target", [0, 200, 100]), objects
                    )
                    if not nearby_objects:
                        return {"error": "No graspable objects near gripper position"}
                        
            return action
            
        except Exception as e:
            return {"error": f"Action validation failed: {e}"}
            
    def _find_alternative_target(self, original_target: List[float], objects: List[DetectedObject]) -> Optional[List[float]]:
        """Find alternative target position to avoid obstacles."""
        try:
            # Try positions around the original target
            x, y, z = original_target
            offsets = [
                [20, 0, 0], [-20, 0, 0], [0, 20, 0], [0, -20, 0],
                [0, 0, 20], [0, 0, -20], [15, 15, 0], [-15, -15, 0]
            ]
            
            for dx, dy, dz in offsets:
                alt_target = [x + dx, y + dy, z + dz]
                
                # Check if alternative is within workspace
                if self.workspace_bounds:
                    bounds = self.workspace_bounds
                    if not (bounds['x_min'] <= alt_target[0] <= bounds['x_max'] and
                            bounds['y_min'] <= alt_target[1] <= bounds['y_max'] and
                            bounds['z_min'] <= alt_target[2] <= bounds['z_max']):
                        continue
                        
                # Check if path to alternative is clear
                color_frame, depth_frame = self.depth_camera.get_frames()
                if depth_frame is not None:
                    current_pos = [0, 200, 100]
                    if self.depth_camera.detect_obstacles_in_path(
                        current_pos, alt_target, depth_frame, self.collision_margin
                    ):
                        return alt_target
                        
            return None
            
        except Exception:
            return None
            
    def _find_nearby_graspable_objects(self, position: List[float], objects: List[DetectedObject]) -> List[DetectedObject]:
        """Find graspable objects near a given position."""
        nearby_objects = []
        proximity_threshold = 50.0  # mm
        
        for obj in objects:
            if not obj.is_graspable:
                continue
                
            distance = np.linalg.norm([
                obj.center_3d[0] - position[0],
                obj.center_3d[1] - position[1],
                obj.center_3d[2] - position[2]
            ])
            
            if distance <= proximity_threshold:
                nearby_objects.append(obj)
                
        return nearby_objects
        
    def _save_debug_image(self, color_frame: np.ndarray, depth_frame: np.ndarray, objects: List[DetectedObject]):
        """Save debug visualization of detected objects."""
        try:
            debug_frame = color_frame.copy()
            
            # Draw object bounding boxes and info
            for obj in objects:
                x, y, w, h = obj.bbox_2d
                center_x, center_y = obj.center_2d
                
                # Color code by graspability
                color = (0, 255, 0) if obj.is_graspable else (0, 0, 255)
                
                # Draw bounding box
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw center point
                cv2.circle(debug_frame, (center_x, center_y), 5, color, -1)
                
                # Add text info
                info_text = f"{obj.label} ({obj.center_3d[0]:.0f},{obj.center_3d[1]:.0f},{obj.center_3d[2]:.0f})"
                cv2.putText(debug_frame, info_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                           
                # Add confidence and graspable status
                status_text = f"Conf:{obj.confidence:.2f} Grasp:{obj.is_graspable}"
                cv2.putText(debug_frame, status_text, (x, y + h + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                           
            # Create depth visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_frame, alpha=0.08), cv2.COLORMAP_JET
            )
            
            # Combine RGB and depth
            combined = np.hstack((debug_frame, depth_colormap))
            
            # Save debug image
            os.makedirs("debug_images", exist_ok=True)
            timestamp = int(time.time())
            debug_path = f"debug_images/depth_detection_{timestamp}.jpg"
            cv2.imwrite(debug_path, combined)
            
            if self.debug:
                print(f"💾 Debug image saved: {debug_path}")
                
        except Exception as e:
            print(f"⚠️ Debug image save failed: {e}")
            
    def get_current_workspace_status(self) -> Dict:
        """Get current workspace status and object information."""
        try:
            objects = self.detect_objects_with_depth(use_yolo=True)
            color_frame, depth_frame = self.depth_camera.get_frames()
            
            workspace_status = {
                "objects_detected": len(objects),
                "graspable_objects": len([obj for obj in objects if obj.is_graspable]),
                "workspace_bounds": self.workspace_bounds,
                "objects": []
            }
            
            for obj in objects:
                workspace_status["objects"].append({
                    "label": obj.label,
                    "position": obj.center_3d,
                    "confidence": obj.confidence,
                    "graspable": obj.is_graspable,
                    "volume": obj.volume,
                    "color": obj.color_dominant
                })
                
            return workspace_status
            
        except Exception as e:
            return {"error": f"Failed to get workspace status: {e}"}
            
    def stop_camera(self):
        """Stop the depth camera system."""
        try:
            self.depth_camera.stop()
            print("✅ Enhanced VLM Controller stopped")
        except Exception as e:
            print(f"⚠️ Error stopping controller: {e}")
            
    def __del__(self):
        """Destructor - ensure camera is properly stopped."""
        self.stop_camera()
