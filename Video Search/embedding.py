import torch
from torchvision import transforms
from PIL import Image
from autoencoder import ConvAutoencoder
import cv2

def object_embedding(img):
    model = ConvAutoencoder()
    model.load_state_dict(torch.load('autoencoder.pth'))
    model.eval()

    # img = Image.open('temp.jpg')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = img.to(device)
    model = model.to(device)

    with torch.no_grad():
        output,embedding = model(img)
        embedding=embedding.cpu().numpy()

    return embedding