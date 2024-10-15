import cv2
import os
import re

def sanitize_folder_name(folder_name):
    # Replace any invalid characters with underscores
    return re.sub(r'[<>:"/\\|?*]', '_', folder_name)

def extract_frames_from_videos(video_folder, output_folder, fps=5):
    # Ensure base output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # List all video files in the directory and its subdirectories
    video_files = []
    for root, dirs, files in os.walk(video_folder):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add other extensions if needed
                video_files.append(os.path.join(root, file))

    # Check if any video files were found
    if not video_files:
        print(f"No video files found in {video_folder}.")
        return

    # Loop through all videos
    for video_file in video_files:
        video_path = video_file
        
        # Create a subfolder for each video to store its frames
        video_name = sanitize_folder_name(os.path.splitext(os.path.basename(video_file))[0])
        video_output_folder = os.path.join(output_folder, video_name)

        # Create the output folder for the video, including parent directories if necessary
        try:
            os.makedirs(video_output_folder, exist_ok=True)
        except FileNotFoundError as e:
            print(f"Error creating directory {video_output_folder}: {e}")
            continue

        # Extract frames from each video
        cap = cv2.VideoCapture(video_path)
        count = 0
        frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Frame rate of the video
        
        # Check if frame_rate is valid
        if frame_rate == 0 or frame_rate is None:
            print(f"Warning: Could not determine frame rate for {video_file}. Skipping...")
            cap.release()
            continue
        
        success, image = cap.read()
        
        while success:
            if count % int(frame_rate / fps) == 0:
                # Save frame as an image
                frame_filename = os.path.join(video_output_folder, f"frame_{count}.jpg")
                cv2.imwrite(frame_filename, image)
                print(f"Saved frame {count} from {video_file}")
                
            success, image = cap.read()
            count += 1

        cap.release()
        print(f"Finished extracting frames for {video_file}")

# Example usage:
video_folder = "C:\\SignLanguage\\Dataset"
output_folder = "C:\\SignLanguage\\OutputFolder"
extract_frames_from_videos(video_folder, output_folder, fps=5)
