import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import csv

torch.manual_seed(42)


# 定义自己的数据集类 Define the dataloader
class MyDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.label_df = pd.read_csv(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        img_name = f"{self.label_df.iloc[idx, 0]}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        label = self.label_df.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label


# 定义卷积神经网络 Design the CNN
torch.manual_seed(42)
CHANNELS = []


class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=3,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        Fx = self.conv1(x)
        Fx = self.bn1(Fx)
        Fx = self.relu1(Fx)
        Fx = self.conv2(Fx)
        Fx = self.bn2(Fx)
        x = Fx + identity
        x = self.relu2(x)

        return x


class ResNet(nn.Module):
    def __init__(self, class_size):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpooling = nn.MaxPool2d(kernel_size=3,stride = 2,padding = 1)

        self.in_channels = 8
        # self.channel = channel
        self.class_size = class_size
        # self.n_blocks = len(channel) - 1

        self.layer1 = self.make_layers(1, 8)
        self.layer2 = self.make_layers(1, 16, stride=2)
        self.layer3 = self.make_layers(1, 16, stride=1)
        self.layer4 = self.make_layers(1, 32, stride=2)
        self.FC = nn.Linear(32, 3)

    def make_layers(self, blocks, out_channels, stride=1):
        layer = []

        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, stride=stride, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)

            )
        layer.append(
            Resblock(in_channels=self.in_channels, out_channels=out_channels, stride=stride, downsample=downsample))
        for i in range(1, blocks):
            layer.append(Resblock(in_channels=out_channels, out_channels=out_channels))
        self.in_channels = out_channels
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpooling(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        compress = nn.AdaptiveAvgPool2d(1)
        x = compress(x).reshape(x.shape[0], -1)
        x = self.FC(x)
        return x


# 定义自己的数据集类 Define the dataloader
class MyDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.label_df = pd.read_csv(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        index = int(self.label_df.iloc[idx, 0])
        img_name = f"{self.label_df.iloc[idx, 0]}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        label = self.label_df.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label


torch.manual_seed(42)
CHANNELS = []


class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=3,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        Fx = self.conv1(x)
        Fx = self.bn1(Fx)
        Fx = self.relu1(Fx)
        Fx = self.conv2(Fx)
        Fx = self.bn2(Fx)
        x = Fx + identity
        x = self.relu2(x)

        return x


class ResNet(nn.Module):
    def __init__(self, class_size):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpooling = nn.MaxPool2d(kernel_size=3,stride = 2,padding = 1)

        self.in_channels = 8
        # self.channel = channel
        self.class_size = class_size
        # self.n_blocks = len(channel) - 1

        self.layer1 = self.make_layers(1, 8)
        self.layer2 = self.make_layers(1, 16, stride=2)
        self.layer3 = self.make_layers(1, 16, stride=1)
        self.layer4 = self.make_layers(1, 32, stride=2)
        self.FC = nn.Linear(32, 3)

    def make_layers(self, blocks, out_channels, stride=1):
        layer = []

        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, stride=stride, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)

            )
        layer.append(
            Resblock(in_channels=self.in_channels, out_channels=out_channels, stride=stride, downsample=downsample))
        for i in range(1, blocks):
            layer.append(Resblock(in_channels=out_channels, out_channels=out_channels))
        self.in_channels = out_channels
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpooling(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        compress = nn.AdaptiveAvgPool2d(1)
        x = compress(x).reshape(x.shape[0], -1)
        x = self.FC(x)
        return x


# 定义自己的数据集类 Define the dataloader


if __name__ == '__main__':
    # -------------读取训练集,训练集地址已经设定好，下面这段不用修改------------------#
    # -----Read the training set, the address of the training set has been set, and the following section does not need to be modified-------#
    #train_path = "/bohr/train-jcym/v1/"
    train_path = "D:/AI4S_Teen_Cup_2025/dataset/Biology/"  # "/bohr/train-jcym/v1/"
    # train_path = "logy/" # "/bohr/train-jcym/v1/"

    # -------------读取测试集---------------#“DATA_PATH”是测试集加密后的环境变量，按照如下方式可以在提交后，系统评分时访问测试集，但是选手无法直接下载
    # ----Read the testing set, “DATA_PATH” is an environment variable for the encrypted test set. After submission, you can access the test set for system scoring in the following manner, but the contestant cannot download it directly.-----#
    if os.environ.get('DATA_PATH'):
        DATA_PATH = os.environ.get("DATA_PATH") + "/"
    else:
        print("Baseline运行时，因为无法读取测试集，所以会有此条报错，属于正常现象")
        print(
            "When baseline is running, this error message will appear because the test set cannot be read, which is a normal phenomenon.")
        # Baseline运行时，因为无法读取测试集，所以会有此条报错，属于正常现象
        # When baseline is running, this error message will appear because the test set cannot be read, which is a normal phenomenon.
    # 数据预处理

    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((80, 60)),
        transforms.ToTensor(),
    ])
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((80, 60)),
        transforms.ToTensor(),
    ])
    # -----------使用自己定义的DataLoader读取数据----------#

    # train_dataset = MyDataset(img_dir=train_path + 'image_train', label_file=train_path + 'label_train_crop.csv',
    #                          transform=transform)

    train_dataset = MyDataset(img_dir=train_path + 'image_train', label_file=train_path + 'label_train.csv',
                              transform=transform_train)

    #test_dataset = MyDataset(img_dir=DATA_PATH + 'image_test', label_file=DATA_PATH + 'label_test_nolabel.csv',
    #                         transform=transform)

    test_dataset = train_dataset
    train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    # --------------------开始训练模型 Start Training and Testing---------------------#
    net = ResNet(3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.1)

    # 训练网络  Training
    num_epochs = 220
    if os.path.exists("net.pth"):
        checkpoint = torch.load("net.pth", weights_only=True)
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_static_dict'])
        net.load_state_dict(checkpoint["model_static_dict"])

    for epoch in range(num_epochs):
        print("epoch:", epoch)
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_accuracy = correct / total
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Accuracy: {epoch_accuracy}")
        if epoch % 10 == 9:
            print(f" epoc:{epoch} Loss: {running_loss / len(train_loader)}, Accuracy: {epoch_accuracy}")
            torch.save({
                'epoch': epoch,
                'model_static_dict': net.state_dict(),
                'optimizer_static_dict': optimizer.state_dict(),
            }, "net.pth")
    # 创建一个空的DataFrame来存储图片名称和预测的label值，Create an empty DataFrame to store image names and predicted label values.
    submission_df = pd.DataFrame(columns=['file_id', 'label'])  # 遍历图片 Traverse images
    file_name_mapping = {i: f"{row[0]}.jpg" for i, row in enumerate(test_dataset.label_df.itertuples(index=False))}
    # print(file_name_mapping)
    net.eval()  # Test
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().numpy()
            batch_size = images.size(0)
            batch_indices = list(range(i * batch_size, (i + 1) * batch_size))
            batch_file_names = [file_name_mapping[idx].replace('.jpg', '') for idx in batch_indices]
            batch_df = pd.DataFrame({'file_id': batch_file_names, 'label': predicted})
            submission_df = pd.concat([submission_df, batch_df], ignore_index=True)

    # 根据中学生物学知识，将0,1,2替换为细胞名称，建议下载图片后，先确定对应顺序，以下的对应顺序仅仅是参考，并不准确
    # According to high school biology knowledge, replace 0, 1, 2 with cell names.
    # It is recommended to determine the corresponding order after downloading the image. The following corresponding order is just for reference and may not be accurate.
    submission_df['label'] = submission_df['label'].map(
        {1: 'Dermal Tissue Cell', 2: 'Meristematic Tissue Cell', 0: 'Epidermis Cell'})
    submission_df.to_csv('submission.csv', index=False)