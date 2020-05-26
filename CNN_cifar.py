import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from my_module import Net
import torch.optim as optim
import matplotlib.pyplot as plt


def main():
    # super parameter
    LR = 0.001
    BatchSize = 4
    Epoch = 2

    # data preprocessing
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load train data
    trainset = torchvision.datasets.CIFAR10(root='./data/cifar-10-python', train=True,
                                            download=False, transform=transform)

    trainloader = Data.DataLoader(trainset, batch_size=BatchSize,
                                  shuffle=True, num_workers=2)

    # plot images
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    # imshow(torchvision.utils.make_grid(images))

    # load test data
    testset = torchvision.datasets.CIFAR10(root='./data/cifar-10-python', train=False,
                                           download=False, transform=transform)
    testloader = Data.DataLoader(testset, batch_size=BatchSize,
                                 shuffle=False, num_workers=2)

    # labels sequence
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = Net()  # load module
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    loss_fuc = torch.nn.CrossEntropyLoss()

    # training
    for epoch in range(Epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):

            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_fuc(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    # test the module
    correct = 0
    total = 0
    with torch.no_grad():  # 不用计算每个节点的损失和误差梯度,可减少算力
        for data in testloader:
            images, labels = data
            outputs = net(images)
            predicted = torch.max(outputs.data, 1)[1]  # 1 表示按行比较
            total += labels.size(0)
            correct += sum(predicted == labels)

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    # test every class
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            predicted = torch.max(outputs, 1)[1]
            c = (predicted == labels).squeeze()  # 计算这一批的准确性
            for i in range(BatchSize):
                label = labels[i]
                class_correct[label] += c[i].item()  # 计算这一批每一类的准确性
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    # save trained module
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)


if __name__=='__main__':
    main()