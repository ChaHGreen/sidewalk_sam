import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch
from torchvision.transforms import functional as F
from preprocess import preprocess_video_frames
import os
import glob


# Load the pre-trained model and set it to evaluation mode
def load_pretrained_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

# Function to perform object detections
def detect_objects(frames, model):
    detections = []
    for frame in frames:
        # Convert frame to a PyTorch tensor, add batch dimension, and send to device
        frame_tensor = F.to_tensor(frame).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            prediction = model(frame_tensor)

        # Process prediction results
        frame_detections = process_predictions(prediction)
        detections.append(frame_detections)

    return detections

# Utility function to process model predictions
def process_predictions(predictions):
    """
    Processes prediction results from the model for a single frame.
    
    :param predictions: Model predictions for a single frame.
    :return: Processed predictions in a structured format.
    """
    processed_results = []
    prediction = predictions[0]  # Assuming batch size of 1

    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        result = {
            'bbox': box.tolist(),
            'label': label.item(),
            'score': score.item(),
        }
        processed_results.append(result)

    return processed_results

if __name__ == "__main__":
    # Example usage
    model = load_pretrained_model()

    # Assume `preprocessed_frames` is a list of frames already preprocessed
    # You need to replace this with actual preprocessed frames
    preprocess_video_frames
    preprocessed_frames = [...]  # Placeholder for preprocessed frames
    for i in glob.glob("videos/"+"*.mp4")
    detections = detect_objects(preprocessed_frames, model)
    # print(detections)
