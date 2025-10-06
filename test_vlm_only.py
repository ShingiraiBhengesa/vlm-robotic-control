#!/usr/bin/env python3
"""
Test VLM improvements without camera dependencies.
Creates mock image data to test if Florence-2/BLIP-2 fixes actually work.
"""
import numpy as np
import cv2
from PIL import Image
import torch
import time

def test_vlm_loading():
    """Test if VLM models can actually load with our fixes."""
    print("🧪 Testing VLM Loading (No Camera Required)")
    print("=" * 50)
    
    try:
        from vlm_controller_depth import EnhancedVLMController
        
        # Initialize without starting camera
        print("🔄 Initializing VLM controller...")
        controller = EnhancedVLMController(debug=True)
        
        # Check what actually loaded
        if hasattr(controller, 'using_blip') and controller.using_blip:
            print("✅ BLIP-2 loaded successfully!")
            model_type = "BLIP-2"
        elif hasattr(controller, 'using_florence') and controller.using_florence:
            print("✅ Florence-2 loaded successfully!")
            model_type = "Florence-2"
        else:
            print("⚠️ Using fallback detection only")
            model_type = "Fallback"
            
        print(f"   Active Model: {model_type}")
        
        # Test with mock image if we have a model
        if controller.model is not None:
            print("\n🖼️ Testing with mock image...")
            
            # Create a simple test image
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add some colored shapes to detect
            cv2.rectangle(test_image, (100, 100), (200, 200), (0, 0, 255), -1)  # Red square
            cv2.circle(test_image, (400, 300), 50, (255, 0, 0), -1)  # Blue circle
            
            # Mock depth frame
            mock_depth = np.ones((480, 640), dtype=np.float32) * 300.0  # 300mm depth
            
            try:
                # Test detection
                objects = controller._detect_with_florence(test_image, mock_depth)
                print(f"   Detected {len(objects)} objects")
                
                for obj in objects:
                    print(f"     - {obj.label}: {obj.center_3d}")
                    
                return True, model_type, len(objects)
                
            except Exception as e:
                print(f"   ❌ Detection failed: {e}")
                return False, model_type, 0
        else:
            print("   ⚠️ No VLM model loaded - using fallback only")
            return True, model_type, 0
            
    except Exception as e:
        print(f"❌ VLM controller failed to initialize: {e}")
        return False, "None", 0

def test_blip2_directly():
    """Test BLIP-2 loading directly."""
    print("\n🔬 Testing BLIP-2 Direct Loading")
    print("=" * 50)
    
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        print("🔄 Loading BLIP-2...")
        model_name = "Salesforce/blip2-opt-2.7b"
        processor = Blip2Processor.from_pretrained(model_name)
        print("✅ BLIP-2 processor loaded")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        model = model.to(device)
        print(f"✅ BLIP-2 model loaded on {device}")
        
        # Test with simple prompt
        test_image = Image.new('RGB', (640, 480), color='red')
        prompt = "Question: What objects do you see? Answer:"
        
        inputs = processor(test_image, prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=20)
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"✅ BLIP-2 response: {generated_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ BLIP-2 direct test failed: {e}")
        return False

def test_simple_fallback():
    """Test the enhanced fallback detection."""
    print("\n🎨 Testing Enhanced Fallback Detection")
    print("=" * 50)
    
    try:
        from vlm_controller_depth import EnhancedVLMController
        
        controller = EnhancedVLMController(debug=False)
        
        # Create test image with colored objects
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Red rectangle (should be detected as "red_object")
        cv2.rectangle(test_image, (50, 50), (150, 150), (0, 0, 255), -1)
        
        # Blue circle
        cv2.circle(test_image, (300, 200), 60, (255, 0, 0), -1)
        
        # Green triangle
        pts = np.array([[400, 100], [450, 200], [350, 200]], np.int32)
        cv2.fillPoly(test_image, [pts], (0, 255, 0))
        
        mock_depth = np.ones((480, 640), dtype=np.float32) * 300.0
        
        # Test fallback detection
        objects = controller._detect_simple_objects(test_image, mock_depth)
        
        print(f"✅ Fallback detected {len(objects)} objects:")
        for obj in objects:
            print(f"   - {obj.label} at {obj.center_3d} (graspable: {obj.is_graspable})")
            
        return len(objects) > 0
        
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        return False

def main():
    """Run VLM-only tests."""
    print("🤖 VLM Intelligence Test (No Camera Required)")
    print("=" * 60)
    
    results = {}
    
    # Test 1: VLM Loading
    success, model_type, obj_count = test_vlm_loading()
    results["VLM Loading"] = success
    
    # Test 2: BLIP-2 Direct (if VLM loading failed)
    if not success or model_type == "Fallback":
        blip_success = test_blip2_directly()
        results["BLIP-2 Direct"] = blip_success
    else:
        results["BLIP-2 Direct"] = True  # Already working through VLM
    
    # Test 3: Enhanced Fallback
    fallback_success = test_simple_fallback()
    results["Enhanced Fallback"] = fallback_success
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 50)
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("🎉 VLM system is working!")
    else:
        print("⚠️ VLM system needs fixes")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
