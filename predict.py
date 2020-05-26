from my_module import Net
import torch
import torchvision.transforms as transforms
from PIL import  Image

transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# labels sequence
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

PATH = './cifar_net.pth'
net = Net()
net.load_state_dict(torch.load(PATH))
img = Image.open('./test_image.jpg')
img = transform(img)  # [channel,h,w]
img = torch.unsqueeze(img, dim= 0)  # [B,C,H,W]


# test
with torch.no_grad():
    outputs = net(img)
    predicted = torch.max(outputs.data, 1)[1] # 1 表示按行比较
    print(predicted)
    print(classes[predicted])
