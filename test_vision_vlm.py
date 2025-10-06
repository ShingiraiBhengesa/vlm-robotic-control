#!/usr/bin/env python3
"""
Focused test for vision and VLM components only.
Tests camera, object detection, and basic VLM functionality.
"""
import cv2
import numpy as np
import time
from vision.depth_camera import DepthCamera
from vlm_controller_depth import EnhancedVLMController

def test_camera_basic():
    """Test basic camera functionality."""
    print("🎥 Testing Camera Functionality...")
    print("=" * 50)
    
    camera = DepthCamera(width=640, height=480, fps=30)
    
    if not camera.start():
        print("❌ Failed to start camera")
        return False
        
    print("✅ Camera started successfully")
    
    # Test frame capture
    for i in range(5):
        color_frame, depth_frame = camera.get_frames()
        if color_frame is not None:
            print(f"✅ Frame {i+1}: Color shape {color_frame.shape}")
            if depth_frame is not None:
                print(f"   Depth shape {depth_frame.shape}, non-zero pixels: {np.count_nonzero(depth_frame)}")
            else:
                print("   Depth: None (fallback mode)")
        else:
            print(f"❌ Frame {i+1}: Failed to capture")
        time.sleep(0.5)
        
    camera.stop()
    return True

def test_object_detection():
    """Test object detection functionality."""
    print("\n🔍 Testing Object Detection...")
    print("=" * 50)
    
    try:
        vlm_controller = EnhancedVLMController(debug=True)
        
        if not vlm_controller.start_camera():
            print("❌ Failed to start VLM camera")
            return False
            
        print("✅ VLM controller started")
        
        # Wait for camera to stabilize
        time.sleep(2)
        
        # Test object detection
        print("🔍 Running object detection...")
        detected_objects = vlm_controller.detect_objects_with_depth(use_yolo=False)
        
        if detected_objects:
            print(f"✅ Detected {len(detected_objects)} objects:")
            for i, obj in enumerate(detected_objects, 1):
                print(f"  {i}. {obj.label}")
                print(f"     Position: {obj.center_3d} mm")
                print(f"     Confidence: {obj.confidence:.2f}")
                print(f"     Graspable: {obj.is_graspable}")
                print(f"     Color: {obj.color_dominant}")
        else:
            print("⚠️ No objects detected")
            
        # Test workspace status
        print("\n📊 Testing workspace status...")
        status = vlm_controller.get_current_workspace_status()
        if "error" not in status:
            print(f"✅ Workspace status: {status['objects_detected']} objects, {status['graspable_objects']} graspable")
            if status['workspace_bounds']:
                bounds = status['workspace_bounds']
                print(f"   Bounds: X[{bounds['x_min']:.0f}:{bounds['x_max']:.0f}] Y[{bounds['y_min']:.0f}:{bounds['y_max']:.0f}] Z[{bounds['z_min']:.0f}:{bounds['z_max']:.0f}]")
        else:
            print(f"⚠️ Workspace status error: {status['error']}")
            
        vlm_controller.stop_camera()
        return True
        
    except Exception as e:
        print(f"❌ Object detection test failed: {e}")
        return False

def test_action_planning():
    """Test VLM action planning (simplified)."""
    print("\n🧠 Testing Action Planning...")
    print("=" * 50)
    
    try:
        vlm_controller = EnhancedVLMController(debug=False)
        
        if not vlm_controller.start_camera():
            print("❌ Failed to start VLM camera for action planning")
            return False
            
        print("✅ VLM controller started for action planning")
        
        # Wait for stabilization
        time.sleep(2)
        
        # Get some objects first
        detected_objects = vlm_controller.detect_objects_with_depth(use_yolo=False)
        
        if detected_objects:
            print(f"✅ Found {len(detected_objects)} objects for planning")
            
            # Test simple commands
            test_commands = [
                "scan the workspace",
                "open gripper", 
                "move to red object"
            ]
            
            for cmd in test_commands:
                print(f"\n🎯 Testing command: '{cmd}'")
                try:
                    response = vlm_controller.plan_action_with_depth(cmd)
                    if "error" in response:
                        print(f"   ⚠️ Response: {response['error']}")
                    else:
                        print(f"   ✅ Command: {response.get('command', 'Unknown')}")
                        if 'reasoning' in response:
                            print(f"   💭 Reasoning: {response['reasoning']}")
                except Exception as e:
                    print(f"   ❌ Planning failed: {e}")
                    
        else:
            print("⚠️ No objects found for action planning test")
            
        vlm_controller.stop_camera()
        return True
        
    except Exception as e:
        print(f"❌ Action planning test failed: {e}")
        return False

def test_visual_display():
    """Test visual display of camera feed."""
    print("\n📺 Testing Visual Display...")
    print("=" * 50)
    print("   Press 'q' to quit, 's' to take screenshot")
    
    try:
        camera = DepthCamera(width=640, height=480, fps=30)
        
        if not camera.start():
            print("❌ Failed to start camera for display")
            return False
            
        print("✅ Camera started - showing live feed")
        
        while True:
            color_frame, depth_frame = camera.get_frames()
            
            if color_frame is not None:
                # Show the frame
                cv2.imshow('Enhanced Vision Test', color_frame)
                
                # If we have depth, show it too
                if depth_frame is not None and np.any(depth_frame > 0):
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_frame, alpha=0.08), 
                        cv2.COLORMAP_JET
                    )
                    cv2.imshow('Depth View', depth_colormap)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    cv2.imwrite(f'test_screenshot_{timestamp}.jpg', color_frame)
                    print(f"📸 Screenshot saved: test_screenshot_{timestamp}.jpg")
            else:
                print("⚠️ No frame received")
                time.sleep(0.1)
                
        camera.stop()
        cv2.destroyAllWindows()
        return True
        
    except Exception as e:
        print(f"❌ Visual display test failed: {e}")
        cv2.destroyAllWindows()
        return False

def main():
    """Run focused vision/VLM tests."""
    print("🚀 Enhanced Vision & VLM Test Suite")
    print("=" * 50)
    print("Testing vision and VLM components independently...")
    
    tests = [
        ("Camera Basic", test_camera_basic),
        ("Object Detection", test_object_detection),
        ("Action Planning", test_action_planning),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
        
        if results[test_name]:
            print(f"✅ {test_name} test passed")
        else:
            print(f"❌ {test_name} test failed")
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 50)
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("🎉 All vision/VLM tests passed!")
        
        # Offer visual test
        response = input("\n🎥 Run visual display test? (y/n): ")
        if response.lower() == 'y':
            test_visual_display()
    else:
        print("⚠️ Some tests failed. Vision system needs fixes.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
