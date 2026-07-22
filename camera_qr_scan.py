import cv2
import json
import os
import numpy as np
import time

# -------------------------
# QR Decoder
# -------------------------
qr_detector = cv2.QRCodeDetector()

def decode_qr_from_image(img_array):
    """Try multiple methods to decode QR"""
    # Method 1: Direct detection
    data, bbox, _ = qr_detector.detectAndDecode(img_array)
    if data:
        return data, bbox
    
    # Method 2: Multi-detection
    try:
        ok, decoded_infos, points, _ = qr_detector.detectAndDecodeMulti(img_array)
        if ok and decoded_infos:
            for i, d in enumerate(decoded_infos):
                if d:
                    return d, points[i] if points is not None else None
    except Exception:
        pass
    
    # Method 3: Enhanced grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    data2, bbox2, _ = qr_detector.detectAndDecode(enhanced)
    if data2:
        return data2, bbox2
    
    return None, None

def scan_qr_standalone():
    """Standalone QR scanner with OpenCV window"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        with open('scanned_qr.json', 'w') as f:
            json.dump({'error': 'Camera not available'}, f)
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Optional: Create named window with specific size
    cv2.namedWindow("QR Scanner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("QR Scanner", 800, 600)  # Set window size to 800x600
    
    print("=" * 60)
    print("QR CODE SCANNER")
    print("=" * 60)
    print("Instructions:")
    print("- Hold QR code steady in front of camera")
    print("- Press 'q' to quit without scanning")
    print("=" * 60)
    
    stable_id = None
    stable_count = 0
    min_stable_frames = 2
    scanned = False
    
    while not scanned:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Create display frame
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Try to decode QR
        data, bbox = decode_qr_from_image(frame)
        
        if data:
            # Stability check
            if stable_id == data:
                stable_count += 1
            else:
                stable_id = data
                stable_count = 1
            
            # Draw detection box
            if bbox is not None and len(bbox) > 0:
                pts = bbox[0].astype(int) if len(bbox.shape) == 3 else bbox.astype(int)
                cv2.polylines(display, [pts], True, (0, 255, 0), 3)
            else:
                # Draw full frame green border
                cv2.rectangle(display, (10, 10), (w-10, h-10), (0, 255, 0), 5)
            
            # Show detected ID and stability
            cv2.putText(display, f"QR Detected: {data}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(display, f"Stability: {stable_count}/{min_stable_frames}", (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # If stable, confirm scan with countdown
            if stable_count >= min_stable_frames:
                # Show confirmation for 5 seconds with countdown
                for countdown in range(5, 0, -1):
                    display_confirm = display.copy()
                    
                    # Success message
                    cv2.putText(display_confirm, "SCAN SUCCESSFUL!", (20, 160), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    
                    # Show the scanned ID prominently
                    cv2.putText(display_confirm, f"Student ID: {data}", (20, 220), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    
                    # Confirmation message with countdown
                    cv2.putText(display_confirm, f"Please verify your ID", (20, 270), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    
                    cv2.putText(display_confirm, f"Closing in {countdown}...", (20, 310), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                    
                    cv2.imshow("QR Scanner", display_confirm)
                    cv2.waitKey(1000)  # Wait 1 second per countdown
                
                # Save result
                with open('scanned_qr.json', 'w') as f:
                    json.dump({'student_id': data, 'timestamp': time.time()}, f)
                
                print(f"\n✅ QR Code Scanned Successfully: {data}")
                scanned = True
                break
        else:
            # No QR detected - reset stability
            stable_id = None
            stable_count = 0
            
            # Draw searching frame with corners
            corner_len = 80
            thickness = 4
            color = (255, 255, 0)  # Yellow
            
            # Top-left corner
            cv2.line(display, (30, 30), (30 + corner_len, 30), color, thickness)
            cv2.line(display, (30, 30), (30, 30 + corner_len), color, thickness)
            # Top-right corner
            cv2.line(display, (w-30, 30), (w-30-corner_len, 30), color, thickness)
            cv2.line(display, (w-30, 30), (w-30, 30+corner_len), color, thickness)
            # Bottom-left corner
            cv2.line(display, (30, h-30), (30+corner_len, h-30), color, thickness)
            cv2.line(display, (30, h-30), (30, h-30-corner_len), color, thickness)
            # Bottom-right corner
            cv2.line(display, (w-30, h-30), (w-30-corner_len, h-30), color, thickness)
            cv2.line(display, (w-30, h-30), (w-30, h-30-corner_len), color, thickness)
            
            # Status text
            cv2.putText(display, "Searching for QR Code...", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
            cv2.putText(display, "Hold QR code steady in frame", (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # Show instruction footer
        cv2.rectangle(display, (0, h-50), (w, h), (50, 50, 50), -1)
        cv2.putText(display, "Press 'Q' to quit", (20, h-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow("QR Scanner", display)
        
        # Check for quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\n❌ Scan cancelled by user")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Scanner closed.")

if __name__ == "__main__":
    scan_qr_standalone()