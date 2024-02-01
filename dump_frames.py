import cv2
import os

# Replace these variables with your actual video file and output folder
input_video_path = "C:/Users/JR13/OneDrive - CEFAS/My onedrive documents/test_django/video/CQ2014-HERRIN-160819_130508-C6H-022-160822_131241_279.MP4"
output_frames_folder = "C:/Users/JR13/OneDrive - CEFAS/My onedrive documents/test_django/stills"


# Create output folder if it doesn't exist
os.makedirs(output_frames_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate the frame interval to extract every nth frame
frame_interval = 500

# Initialize a counter for the frames
frame_count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    # Break the loop if we have reached the end of the video
    if not ret:
        break
    # Check if the current frame is the 100th frame
    if frame_count % frame_interval == 0:
        # Generate the output file path
        output_file = os.path.join(output_frames_folder, f"frame_{frame_count // frame_interval:04d}.png")
        # Save the frame as a PNG file
        cv2.imwrite(output_file, frame)
    # Increment the frame count
    frame_count += 1

# Release the video capture object
cap.release()

print(f"Frames extracted: {frame_count // frame_interval}/{total_frames // frame_interval}")
