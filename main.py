import torch as t
import torchvision as tv
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import prettytable as pt

show = ToPILImage()

# Constant
DATA = 'data/'
CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
EPOCHS = 2
CUDA = t.cuda.is_available()

# Hyperparameter
BATCH_SIZE = 4

# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),  # 轉為Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 正規化
])

# Training Set
trainset = tv.datasets.CIFAR10(
    root=DATA,
    train=True,
    download=True,
    transform=transform
)

trainloader = t.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

# Testing Set
testset = tv.datasets.CIFAR10(
    root=DATA,
    train=False,
    download=True,
    transform=transform
)

testloader = t.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

# Dataset回傳(data, label)形式的資料
(data, label) = trainset[100]
print(CLASSES[label])

# Dataloader是可迭代，將dataset回傳的每一條dat拼成一個batch，提供多執行緒加速優化與Shuffle等功能
# 當Program對dataset的所有data遍歷完一遍後，相對應的Dataloader也完成了一次迭代
dataiter = iter(trainloader)
images, labels = dataiter.next()  # 回傳4張圖片及標籤
# print(' '.join('%11s' % CLASSES[labels[j]] for j in range(BATCH_SIZE)))

# LeNet


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)  # reshape tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
if CUDA:
    net.cuda()

criterion = nn.CrossEntropyLoss()  # 交叉熵損失函數
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(EPOCHS):

    running_loss = 0.0

    print('now running EPOCH %s/%s' % (epoch + 1, EPOCHS))
    for i, data in enumerate(tqdm(trainloader), 0):

        # 输入数据
        inputs, labels = data
        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)

        # 梯度清零
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # 更新参数
        optimizer.step()

        # 打印log信息
        running_loss += loss.data[0]
        if i % 2000 == 1999:  # 每2000个batch打印一下训练状态
            tqdm.write('[EPOCH=%d, BATCH=%5d] loss: %.3f' %
                       (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()  # 一個Batch回傳四張圖片
if CUDA:
    images = images.cuda()
    labels = labels.cuda()

# 計算圖片在每個類別上的分數
outputs = net(Variable(images))
# 得分最高的類別
_, predicted = t.max(outputs.data, 1)

print('預測結果:')
table=pt.PrettyTable()
table.field_names=['Predicted', 'Ground Truth']

for i in range(4):
    table.add_row([CLASSES[labels[i]], CLASSES[predicted[i]]])
print(table)

correct = 0 # 预测正确的图片数
total = 0 # 总共的图片数
for data in testloader:
    images, labels = data
    if CUDA:
        images = images.cuda()
        labels = labels.cuda()
    outputs = net(Variable(images))
    _, predicted = t.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('10000張測試集中準確率為: %d%%' % (100 * correct / total))