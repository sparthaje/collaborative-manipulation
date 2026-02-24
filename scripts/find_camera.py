#!/usr/bin/env python3
"""
Usage:
  python cam_preview.py <index>

Example:
  python cam_preview.py 0
"""

import sys
import cv2


def usage() -> None:
    print("Usage: python cam_preview.py <camera_index>", file=sys.stderr)
    print("Example: python cam_preview.py 0", file=sys.stderr)


def main() -> int:
    if len(sys.argv) != 2:
        usage()
        return 2

    try:
        cam_index = int(sys.argv[1])
    except ValueError:
        print(f"Error: camera_index must be an integer, got: {sys.argv[1]!r}", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Error: could not open camera index {cam_index}", file=sys.stderr)
        # Optional: quickly scan a few indices like your snippet
        print("Quick scan (0..4):")
        for i in range(5):
            tmp = cv2.VideoCapture(i)
            print(i, tmp.isOpened())
            tmp.release()
        return 1

    # Query width/height (may be 0 if backend hasn't started; we'll also print first-frame size)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"Camera {cam_index} reported size: {width}x{height}")

    window_name = f"Camera {cam_index} (press q or ESC to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    printed_frame_size = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Warning: failed to read frame")
                break

            if not printed_frame_size:
                h, w = frame.shape[:2]
                print(f"First frame actual size: {w}x{h}")
                printed_frame_size = True

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
