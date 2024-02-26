from preprocess import preprocess_video
from detect_objects import detect_objects
from autoencoder import train_autoencoder
from database import store_embeddings

def main():
    # Step 1: Preprocess the video
    preprocessed_frames = preprocess_video("path/to/your/video.mp4", (224, 224))
    
    # Step 2: Detect objects
    detected_objects = []
    for frame in preprocessed_frames:
        objects = detect_objects(frame)
        detected_objects.extend(objects)
    
    # Step 3: Train autoencoder
    autoencoder_model = train_autoencoder(preprocessed_frames)
    
    # Step 4: Store embeddings in database
    store_embeddings(detected_objects)
    
if __name__ == "__main__":
    main()
