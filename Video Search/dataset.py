from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection
import random


class CocoObjectsCropDataset(CocoDetection):
    def __init__(self, root, annFile, classes_of_interest, subset_size=None, transform=None):
        super(CocoObjectsCropDataset, self).__init__(root, annFile, transform)
        self.coco = COCO(annFile)
        self.classes_of_interest = classes_of_interest
        self.image_ids = self._get_relevant_image_ids(subset_size)
        self.ann_ids = self._get_all_annotations_ids()
        print(f"Initialized dataset with {len(self.image_ids)} images and {len(self.ann_ids)} annotations")

    def _get_relevant_image_ids(self, subset_size=None):
        ids = []
        for i, class_name in enumerate(self.classes_of_interest):
            catIds = self.coco.getCatIds(catNms=[class_name])
            imgIds = self.coco.getImgIds(catIds=catIds)
            ids.extend(imgIds)
            print(f"Processed class {class_name}: {len(imgIds)} images")
        
        # if subset_size is define, pick a subset of that size
        if subset_size is not None:
            ids = random.sample(ids, min(subset_size, len(ids)))
        
        return list(set(ids))

    def _get_all_annotations_ids(self):
        ann_ids = []
        for img_id in self.image_ids:
            for class_name in self.classes_of_interest:
                catIds = self.coco.getCatIds(catNms=[class_name])
                annIds = self.coco.getAnnIds(imgIds=img_id, catIds=catIds, iscrowd=None)
                ann_ids.extend(annIds)
        return ann_ids

    def __len__(self):
        return len(self.ann_ids)

    def __getitem__(self, index):
        if index % 100 == 0:  # 每100个图像打印一次进度
            print(f"Loading image {index + 1}/{len(self.ann_ids)}")
        ann_id = self.ann_ids[index]
        annotation = self.coco.loadAnns(ann_id)[0]
        img_id = annotation['image_id']
        coco_img = self.coco.loadImgs(img_id)[0]
        image = Image.open(os.path.join(self.root, coco_img['file_name'])).convert('RGB')

        bbox = annotation['bbox']
        image = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

        if self.transform is not None:
            image = self.transform(image)

        return image

def crop_objects(video_path, detections_csv_path):
    # 使用视频处理函数获取所有帧的信息
    frame_infos = process_video_frames(video_path)
    
    # 读取检测信息
    detections_df = pd.read_csv(detections_csv_path)
    
    cropped_images = []  # 用于存储裁剪的图像
    indexes = []  # 用于存储与裁剪图像相对应的索引
    
    # 遍历每一帧
    for frame_info in frame_infos:
        frame = frame_info['frame']
        frameNum = frame_info['frameNum']
        
        # 获取当前帧的所有检测
        frame_detections = detections_df[detections_df['frameNum'] == frameNum]
        
        for _, det in frame_detections.iterrows():
            # 解析边界框信息，假设bbox格式是"x_min,y_min,x_max,y_max"
            bbox = list(map(int, det['bbox info'].split(',')))
            
            # 裁剪检测到的对象
            crop_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            # 将裁剪的图像添加到列表中
            cropped_images.append(crop_img)

            # In

            index = f"{det['vidId']}_{det['frameNum']}_{det['detectedObjId']}"
            indexes.append(index)

    return cropped_images, indexes

    ## return be like this
    ### cropped_images[i] --> indexes[i] "vidId_frameNum_detectedObjId"  

