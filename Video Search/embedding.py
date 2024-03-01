import torch
from torchvision import transforms
from PIL import Image
from autoencoder import ConvAutoencoder
import cv2
from torchvision.transforms.functional import to_pil_image
import numpy as np

def object_embedding(cropped_img_np):
    # 初始化模型
    model = ConvAutoencoder()
    model.load_state_dict(torch.load('checkpoint/autoencoder.pth', map_location=torch.device('cpu')))
    model.eval()

    # 确保传入的裁剪图像是 numpy 数组
    if not isinstance(cropped_img_np, np.ndarray):
        raise ValueError("The cropped image must be a numpy array")

    # 将 numpy 数组转换为 PIL 图像
    cropped_img = to_pil_image(cropped_img_np)

    # 定义图像的转换操作
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小以匹配模型输入
        transforms.ToTensor(),          # 将 PIL 图像转换为张量
    ])

    # 应用转换
    img_tensor = transform(cropped_img).unsqueeze(0)  # 添加批次维度

    # 指定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)
    model.to(device)

    # 生成嵌入向量
    with torch.no_grad():
        _,embedding = model(img_tensor)  # 假设模型直接返回嵌入向量

    # 将嵌入向量从 GPU 移动到 CPU 并转换为 numpy 数组
    embedding_np = embedding.cpu().numpy().flatten()

    return embedding_np