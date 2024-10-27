import os
import cv2


def extract_frames(
    video_path, output_folder, frame_ratio, width=None, height=None
) -> int:
    # Check if output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    save_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Decide whether to save this frame
        if frame_count % int(1 / frame_ratio) == 0:
            # If width and height are specified, crop the frame
            if width is not None and height is not None:
                h, w, _ = frame.shape
                startx = w // 2 - width // 2
                starty = h // 2 - height // 2
                frame = frame[starty : starty + height, startx : startx + width]

            # Save frame
            cv2.imwrite(
                os.path.join(output_folder, f"frame.{save_count:04d}.png"), frame
            )
            save_count += 1
        frame_count += 1

    cap.release()
    print(f"Finished: Extracted {save_count} frames to {output_folder}")
    return save_count
