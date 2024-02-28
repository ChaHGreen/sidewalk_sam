def write_embeddings_to_csv(csv_path, indexes, embeddings):
    # 读取CSV文件为DataFrame
    df = pd.read_csv(csv_path)
    
    # 遍历每个索引和其对应的嵌入向量
    for index, embedding in zip(indexes, embeddings):
        # 解析索引以获取vidId, frameNum, 和 detectedObjId
        vidId, frameNum, detectedObjId = index.split('_')
        
        # 找到对应的行
        row_index = df[(df['vidId'] == vidId) & (df['frameNum'] == int(frameNum)) & (df['detectedObjId'] == int(detectedObjId))].index
        
        # 假设你要将嵌入向量作为一个新列"embedding"添加到DataFrame中
        # 这里简化处理，将embedding转换为字符串形式存储
        df.at[row_index, 'embedding'] = str(embedding.tolist())
    
    # 将更新后的DataFrame写回CSV文件
    df.to_csv(csv_path, index=False)



import torch
from torchvision import transforms
from PIL import Image
from autoencoder import ConvAutoencoder
import cv2

# 假设 ConvAutoencoder 是你的模型类
model = ConvAutoencoder()
model.load_state_dict(torch.load('autoencoder.pth'))
model.eval()  # 设置为评估模式

# 加载一张图片来进行推断
img = Image.open('temp.jpg')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
img = transform(img).unsqueeze(0)  # 添加批次维度

# 使用 GPU 进行推断（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img = img.to(device)
model = model.to(device)

with torch.no_grad():  # 不计算梯度
    output,embedding = model(img)
    embedding=embedding.cpu()
    print(embedding)
    print(embedding.shape)