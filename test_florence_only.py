#!/usr/bin/env python3
"""
Test ONLY Florence-2 for real object detection.
No BLIP, no other models - just Florence-2 detecting real objects.
"""
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import numpy as np
import cv2

def test_florence2_loading():
    """Test if Florence-2 loads without flash_attn issues."""
    print("🤖 Testing Florence-2 Loading (ONLY)")
    print("=" * 50)
    
    try:
        model_name = "microsoft/Florence-2-base"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔄 Loading Florence-2 on {device}...")
        
        # Try to load Florence-2 with transformers 4.37.0 (auto-accept custom code)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(device)
        
        print("✅ Florence-2 loaded successfully!")
        return True, processor, model, device
        
    except Exception as e:
        print(f"❌ Florence-2 loading failed: {e}")
        return False, None, None, None

def test_florence2_object_detection():
    """Test Florence-2 with real object detection."""
    print("\n🔍 Testing Florence-2 Object Detection")
    print("=" * 50)
    
    # First load Florence-2
    success, processor, model, device = test_florence2_loading()
    if not success:
        return False, []
    
    try:
        # Try with camera first
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("📷 Using camera for real object detection...")
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                
                # Save for inspection
                image.save("florence_test_image.jpg")
                print("💾 Test image saved as florence_test_image.jpg")
                
            else:
                print("❌ Failed to capture frame")
                return False
        else:
            print("⚠️ No camera available, using test image...")
            # Create a simple test image
            test_array = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray background
            # Add some simple shapes that might be detected as objects
            cv2.rectangle(test_array, (100, 100), (300, 300), (255, 255, 255), -1)  # White rectangle
            cv2.circle(test_array, (500, 200), 80, (0, 0, 0), -1)  # Black circle
            
            image = Image.fromarray(test_array)
        
        print("🔄 Running Florence-2 object detection...")
        
        # Use object detection task
        prompt = "<OD>"  # Object Detection task
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        
        # Generate detection
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                early_stopping=False,
                do_sample=False
            )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        print(f"✅ Florence-2 raw response: {generated_text}")
        
        # Parse the response
        parsed_answer = processor.post_process_generation(
            generated_text, 
            task="<OD>", 
            image_size=(image.width, image.height)
        )
        
        if '<OD>' in parsed_answer:
            od_results = parsed_answer['<OD>']
            print(f"✅ Florence-2 detected {len(od_results['labels'])} objects:")
            
            for i, (bbox, label) in enumerate(zip(od_results['bboxes'], od_results['labels'])):
                print(f"   {i+1}. {label} at {bbox}")
                
            # These should be REAL object names like "cup", "person", "bottle"
            # NOT fake names like "red_object", "blue_object"
            return True, od_results['labels']
        else:
            print("⚠️ No objects detected by Florence-2")
            return True, []
            
    except Exception as e:
        print(f"❌ Florence-2 detection failed: {e}")
        return False, []

def main():
    """Test Florence-2 ONLY."""
    print("🚀 Florence-2 ONLY Test")
    print("=" * 40)
    print("Testing if Florence-2 detects REAL objects:")
    print("- cup, bottle, phone, person, etc.")
    print("- NOT fake colored objects!")
    print("=" * 40)
    
    # Test Florence-2 detection
    success, detected_labels = test_florence2_object_detection()
    
    if success:
        if detected_labels:
            print(f"\n✅ SUCCESS: Florence-2 detected {len(detected_labels)} real objects!")
            print("Detected object types:")
            for label in set(detected_labels):  # Remove duplicates
                print(f"  - {label}")
                
            # Check if these are real object names
            fake_names = ['red_object', 'blue_object', 'green_object', 'yellow_object']
            real_detection = not any(fake in detected_labels for fake in fake_names)
            
            if real_detection:
                print("\n🎉 EXCELLENT: Florence-2 is detecting REAL objects!")
                print("   No fake colored objects found")
            else:
                print("\n⚠️ WARNING: Still detecting fake colored objects")
                print("   This means the system is still using fallback detection")
        else:
            print("\n⚠️ Florence-2 loaded but detected no objects")
            print("   Try pointing camera at common objects like cups, phones, etc.")
            
        return success
    else:
        print("\n❌ FAILED: Florence-2 is not working")
        print("   The system will fall back to fake colored object detection")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
