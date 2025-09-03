import cv2
import os
import platform

def take_photo(cam_index):
    save_dir = os.path.join(".", "data", "test")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Detect OS
    system = platform.system()
    if system == "Windows":
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    else:  # Linux, macOS, etc.
        cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print("Error: Cannot access the camera")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        cap.release()
        return

    photo_path = os.path.join(save_dir, "photo.png")
    cv2.imwrite(photo_path, frame)
    print(f"Photo saved at {photo_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cam_index = 2
    take_photo(cam_index)
