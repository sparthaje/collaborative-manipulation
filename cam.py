import cv2
import glob

print("=== Serial / USB Devices ===")
for dev in sorted(glob.glob("/dev/cu.*") + glob.glob("/dev/tty.*")):
    print(dev)

print("\n=== OpenCV Camera Indices ===")
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Index {i} -> OPEN")
        cap.release()
    else:
        print(f"Index {i} -> closed")
