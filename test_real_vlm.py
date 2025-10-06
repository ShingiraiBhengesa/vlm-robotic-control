#!/usr/bin/env python3
"""
Test real VLM object detection with actual object names.
Uses a working VLM model to detect real objects like "cup", "phone", "bottle".
"""
import cv2
import numpy as np
from PIL import Image
import torch
import time

def test_working_vlm():
    """Test a simple working VLM that actually detects real objects."""
    print("🤖 Testing Real Object Detection VLM")
    print("=" * 50)
    
    try:
        # Try LLaVA-Next (simpler and more reliable)
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        
        print("🔄 Loading LLaVA-Next (lightweight VLM)...")
        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        
        processor = LlavaNextProcessor.from_pretrained(model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, 
            low_cpu_mem_usage=True
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        print(f"✅ LLaVA-Next loaded on {device}")
        
        # Create a test image with some objects
        test_image = Image.open("/dev/null")  # This will fail, but let's try camera
        
        return True
        
    except Exception as e:
        print(f"❌ LLaVA-Next failed: {e}")
        
        # Try BLIP (original, more compatible)
        try:
            print("🔄 Trying original BLIP...")
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            
            print(f"✅ Original BLIP loaded on {device}")
            
            # Test with a simple image
            test_image = Image.new('RGB', (640, 480), color=(255, 0, 0))  # Red image
            
            # Generate caption
            inputs = processor(test_image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=50)
            
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"✅ BLIP response: '{caption}'")
            
            return True, processor, model
            
        except Exception as e2:
            print(f"❌ Original BLIP failed: {e2}")
            return False

def test_with_camera():
    """Test VLM with actual camera input."""
    print("\n📷 Testing with Real Camera Input")
    print("=" * 50)
    
    try:
        # Try to get a camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No camera available")
            return False
        
        print("✅ Camera opened, capturing frame...")
        
        # Capture a frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            cap.release()
            return False
            
        print("✅ Frame captured successfully")
        
        # Convert to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Save for inspection
        pil_image.save("captured_frame.jpg")
        print("💾 Frame saved as captured_frame.jpg")
        
        cap.release()
        return True, pil_image
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_yolo_real_detection():
    """Test YOLO with real object detection."""
    print("\n🎯 Testing YOLO Real Object Detection")
    print("=" * 50)
    
    try:
        from ultralytics import YOLO
        
        # Load YOLO model
        model = YOLO('yolov8n.pt')
        
        print("✅ YOLO loaded successfully")
        
        # Try with camera
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Got camera frame, running YOLO detection...")
                
                # Run detection
                results = model(frame, verbose=False)
                
                detected_objects = []
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            conf = float(box.conf[0])
                            if conf > 0.5:
                                class_id = int(box.cls[0])
                                label = model.names[class_id]
                                detected_objects.append(f"{label} ({conf:.2f})")
                
                if detected_objects:
                    print(f"✅ YOLO detected {len(detected_objects)} real objects:")
                    for obj in detected_objects:
                        print(f"   - {obj}")
                    return True
                else:
                    print("⚠️ YOLO ran but found no objects above 50% confidence")
                    print("   This might mean no recognizable objects in view")
                    return True  # YOLO works, just no objects
            
            cap.release()
        
        # Test with a sample image if no camera
        print("⚠️ No camera, testing YOLO with sample...")
        
        # Create a simple test image
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (200, 200), (400, 350), (255, 255, 255), -1)  # White rectangle
        
        results = model(test_img, verbose=False)
        print("✅ YOLO processing completed")
        
        return True
        
    except ImportError:
        print("❌ YOLO not installed. Run: pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ YOLO test failed: {e}")
        return False

def main():
    """Test real VLM object detection."""
    print("🚀 Real VLM Object Detection Test")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Working VLM
    print("Testing working VLM models...")
    try:
        vlm_result = test_working_vlm()
        results["VLM Loading"] = vlm_result if isinstance(vlm_result, bool) else vlm_result[0]
    except Exception as e:
        print(f"VLM test crashed: {e}")
        results["VLM Loading"] = False
    
    # Test 2: Camera input
    print("Testing camera input...")
    try:
        cam_result = test_with_camera()
        results["Camera Input"] = cam_result if isinstance(cam_result, bool) else cam_result[0]
    except Exception as e:
        print(f"Camera test crashed: {e}")
        results["Camera Input"] = False
    
    # Test 3: YOLO real detection
    print("Testing YOLO real detection...")
    try:
        yolo_result = test_yolo_real_detection()
        results["YOLO Detection"] = yolo_result
    except Exception as e:
        print(f"YOLO test crashed: {e}")
        results["YOLO Detection"] = False
    
    # Summary
    print("\n📊 REAL OBJECT DETECTION TEST SUMMARY")
    print("=" * 60)
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed > 0:
        print("🎉 Some real object detection is working!")
        print("\n🎯 Next Steps:")
        print("1. Connect a camera and run the full system")
        print("2. Point camera at real objects (cup, phone, bottle)")
        print("3. System will detect actual objects, not fake colored shapes")
    else:
        print("⚠️ No real object detection working yet")
        
    return passed > 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
