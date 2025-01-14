import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.transforms import functional, InterpolationMode
import os
from PIL import Image
import numpy as np
from model_opt import *
import matplotlib.pyplot as plt


def voc_rand_crop(image, label):
    crop_h, crop_w = model_opt["img_size"]
    w, h = image.size
    pad_h = max(crop_h - h, 0)  # 高度需要填充的像素
    pad_w = max(crop_w - w, 0)  # 宽度需要填充的像素
    padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]
    image = functional.pad(image, padding, fill=0)
    label = functional.pad(label, padding, fill=255)
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=model_opt["img_size"])
    image = functional.crop(image, i, j, h, w)
    label = functional.crop(label, i, j, h, w)
    return image, label


image_transform = transforms.Compose([
    transforms.Resize(model_opt["img_size"]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

label_transform = transforms.Compose([
    transforms.Resize(model_opt["img_size"], interpolation=InterpolationMode.NEAREST)
])


class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.data_path = os.path.join(self.root, 'JPEGImages')
        self.label_path = os.path.join(self.root, 'SegmentationClass')
        self.label_list = os.listdir(self.label_path)

    def __getitem__(self, index):
        label_name = self.label_list[index]
        label_path = os.path.join(self.label_path, label_name)
        image_path = os.path.join(self.data_path, label_name.replace('png', 'jpg'))
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('P')
        image, label = voc_rand_crop(image, label)
        image = image_transform(image)
        label = label_transform(label)
        label = np.array(label)
        label = torch.tensor(label, dtype=torch.int64)
        return image, label

    def __len__(self):
        return len(self.label_list)


dataset = MyDataset(root=model_opt["root"], transform=image_transform)
seed = 42
generator = torch.Generator().manual_seed(seed)
train_dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)],
                                           generator=generator)

if __name__ == '__main__':
    # 获取图像和标签
    image_tensor, label_tensor = train_dataset[11]

    # 将图像从 Tensor 转回 PIL 图像
    image = transforms.ToPILImage()(image_tensor)

    # 标签需要转为 numpy 数组再可视化
    label = label_tensor.numpy()

    # 显示图像和标签
    plt.figure(figsize=(10, 5))

    # 原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    # 标签（使用唯一颜色可视化）
    plt.subplot(1, 2, 2)
    plt.imshow(label, cmap='tab20')  # 使用色彩图以便区分类别
    plt.title("Label")
    plt.axis("off")

    # 展示结果
    plt.show()
