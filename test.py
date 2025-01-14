from dataset import *
from CoTAttention import *
import torch.optim as optim
from model_opt import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import jaccard_score, confusion_matrix
from train import mean_iou
from PIL import Image
from SegNet import *
from SegNet_Attention import *

VOC_COLORMAP = [[128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

image_transform = transforms.Compose([
    transforms.Resize(model_opt["img_size"]),
    transforms.ToTensor(),

])


def test():
    device = torch.device(model_opt["device"])
    model_file = None
    if model_opt['attention'] is True:
        model = SegNet_Attention(model_opt['num_classes']).to(device)
        model_file = "SegNet_Attention"
    else:
        model = SegNet(model_opt['num_classes']).to(device)
        model_file = "SegNet"
    model.load_state_dict(torch.load(os.path.join(model_opt['weight_save_path'], model_file, model_opt['model'])))
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    dataloader = DataLoader(test_dataset, batch_size=model_opt['batch_size'], shuffle=False, num_workers=2)
    writer = SummaryWriter(log_dir="./runs")

    model.eval()  # 切换模型到评估模式
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算
        for images, labels in tqdm(dataloader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            # 模型预测
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)  # 获取预测结果
            correct += (predicted == labels).sum().item()
            total += (labels != 255).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total * 100
    iou = jaccard_score(labels.cpu().numpy().flatten(), predicted.cpu().numpy().flatten(), average='macro')
    print(f"avg_loss: {avg_loss:.4f} , accuracy: {accuracy:.2f}% , miou: {iou:.4f}")

    dummy_input = torch.randn(8, 3, model_opt['img_size'][0], model_opt['img_size'][1]).to(device)
    writer.add_graph(model, dummy_input)


def outputImage(image_name):
    image_path = os.path.join("dataset/JPEGImages/", image_name)
    image = Image.open(image_path).convert('RGB')
    image = image_transform(image)
    image = image.unsqueeze(0)
    device = torch.device(model_opt["device"])
    model_file = None
    if model_opt['attention'] is True:
        model = SegNet_Attention(model_opt['num_classes']).to(device)
        model_file = "SegNet_Attention"
    else:
        model = SegNet(model_opt['num_classes']).to(device)
        model_file = "SegNet"
    model.load_state_dict(torch.load(os.path.join(model_opt['weight_save_path'], model_file, model_opt['model'])))
    print(image.shape)
    image = image.to(device)
    outputs = model(image)
    print(outputs.shape)
    _, predicted = torch.max(outputs, 1)
    print(predicted.cpu().numpy())
    predicted = predicted.squeeze(0).cpu().numpy()
    # 根据VOC_COLORMAP映射类别标签到颜色
    random_color = VOC_COLORMAP[np.random.randint(len(VOC_COLORMAP))]
    color_image = np.zeros((predicted.shape[0], predicted.shape[1], 3), dtype=np.uint8)
    for i in range(predicted.shape[0]):
        for j in range(predicted.shape[1]):
            label = predicted[i, j]
            if label != 0:  # 排除背景类（0）
                # 所有非背景像素赋予随机颜色
                color_image[i, j] = random_color

    # 将颜色图像转换为PIL图像并保存
    output_image = Image.fromarray(color_image)
    output_image.show()  # 显示图像
    output_image.save(os.path.join(model_opt['predictions_path'], image_name))  # 保存图像


"""
SegNet:
avg_loss: 1.6443 , accuracy: 74.65% , miou: 0.1379
SegNet_Attention:
avg_loss: 1.4488 , accuracy: 73.24% , miou: 0.2548  
"""
