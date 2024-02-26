import cv2
import numpy as np

def preprocess_video_frames(video_path, skip_frames=5, frame_size=(224, 224), scale=True, normalize=True):

    processed_frames = []
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)    ## read the video
    frame_count = 0

    while True:
        ret, frame = cap.read()  ## decode the frame
        if not ret:
            break
        
        if frame_count % skip_frames == 0:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            resized_frame = cv2.resize(frame, frame_size)
            
            if scale:
                resized_frame = resized_frame / 255.0
            
            if normalize:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                # Ensure resized_frame is float before subtraction and division
                resized_frame = (resized_frame - mean) / std
            
            processed_frames.append(resized_frame)
        
        frame_count += 1

    cap.release()
    return processed_frames   ## return RGB format

# Example usage
video_path = 'path/to/your/video.mp4'
processed_frames = preprocess_video_frames(video_path, skip_frames=5, frame_size=(224, 224), scale=True, normalize=True)
