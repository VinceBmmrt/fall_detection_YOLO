"""Scan cv2 indexes 0-9 and print which cameras open successfully."""
import cv2


def find_cameras() -> None:
    found = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            found.append((i, w, h))
            cap.release()

    if not found:
        print("No cameras found.")
    else:
        for idx, w, h in found:
            print(f"  [{idx}] {w}x{h}")


if __name__ == "__main__":
    find_cameras()
