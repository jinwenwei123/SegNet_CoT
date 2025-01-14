from dataset import *
from CoTAttention import *
import torch.optim as optim
from model_opt import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from SegNet import *
from SegNet_Attention import *


def pixel_accuracy(preds, labels, ignore_index=255):
    # preds: 模型的预测结果 (batch_size, num_classes, height, width)
    # labels: 真实标签 (batch_size, height, width)
    # ignore_index: 用于忽略的标签值
    _, predicted = torch.max(preds, 1)  # 获取每个像素的预测类别
    total_pixels = (labels != ignore_index).sum().float()  # 排除掉忽略值
    correct_pixels = (predicted == labels).sum().float()  # 预测正确的像素数
    accuracy = correct_pixels / total_pixels
    return accuracy


def mean_iou(preds, labels, num_classes, ignore_index=255):
    # preds: 模型的预测结果 (batch_size, num_classes, height, width)
    # labels: 真实标签 (batch_size, height, width)
    # num_classes: 类别数量
    # ignore_index: 用于忽略的标签值
    _, predicted = torch.max(preds, 1)  # 获取每个像素的预测类别
    ious = []

    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        # 计算每个类别的 TP, FP, FN
        intersection = ((predicted == cls) & (labels == cls)).sum().float()  # 真阳性
        union = ((predicted == cls) | (labels == cls)).sum().float()  # 并集
        if union == 0:
            ious.append(float('nan'))  # 避免除零错误
        else:
            ious.append(intersection / union)

    # 将 ious 从 GPU 转移到 CPU，再转成 NumPy 数组
    ious = torch.tensor(ious).cpu().numpy()

    # 计算 mIoU，排除 NaN 值
    return np.nanmean(ious)


def train(epoch, attention=False):
    device = torch.device(model_opt["device"])
    # model = UNet(model_opt['num_classes'], attention=model_opt['attention']).to(device)
    if attention is True:
        model = SegNet_Attention(model_opt['num_classes']).to(device)
    else:
        model = SegNet(model_opt['num_classes']).to(device)
    if model_opt['pretrain'] is True:
        if model_opt['attention'] is True:
            model.load_state_dict(torch.load("weight/SegNet_Attention/" + model_opt['model']))
        else:
            model.load_state_dict(torch.load("weight/SegNet/" + model_opt['model']))
    else:
        model.load_state_dict(torch.load(r"weight/vgg16_bn-6c64b313.pth"), strict=False)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=model_opt['lr'])
    dataloader = DataLoader(train_dataset, batch_size=model_opt['batch_size'], shuffle=True, num_workers=2)
    writer = SummaryWriter(log_dir="./runs/train")

    for i in range(epoch):
        model.train()
        loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
        running_loss = 0.0
        total_accuracy = 0
        total_iou = 0
        for idx, (images, labels) in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            accuracy = pixel_accuracy(outputs, labels)
            total_accuracy += accuracy.item()
            iou = mean_iou(outputs, labels, num_classes=model_opt['num_classes'])
            total_iou += iou
            loop.set_description(f'Epoch [{i + 1}/{epoch}]')
            loop.set_postfix(loss=f"{running_loss / (idx + 1):.4f}", acc=f"{total_accuracy / (idx + 1):.4f}",
                             mIoU=f"{total_iou / (idx + 1):.4f}")

        if model_opt['attention'] is True:
            torch.save(model.state_dict(),
                       os.path.join(model_opt['weight_save_path'], f"SegNet_Attention/model_S+A.pth"))
        else:
            torch.save(model.state_dict(),
                       os.path.join(model_opt['weight_save_path'], f"SegNet/model_S.pth"))
        if model_opt['attention'] is True:
            writer.add_scalar("Loss/batch(S_A)", running_loss / len(dataloader), i + 1)
        else:
            writer.add_scalar("Loss/batch(S)", running_loss / len(dataloader), i + 1)


if __name__ == '__main__':
    weights = torch.tensor([0.1] + [1.0] * (model_opt['num_classes'] - 1),
                           device=torch.device(model_opt["device"]))  # 背景权重降低
    print(weights)
