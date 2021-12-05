import matplotlib.pyplot as plt
from IPython import display
import sys
import matplotlib


def use_svg_display():
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser',
                   'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker',
                   'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这里的_表示我们忽略(不使用)的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


def load_data_fashion_mnist(batch_size, resize = None, root='~/Datasets/FashionMNIST'):
    #Download the fashion mnist dataset and thenload into memory.
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_iter, test_iter


def linreg(X, w, b): # 本函数已保存在d2lzh_pytorch 包中方便以后使用
    return torch.mm(X, w) + b

def squared_loss(y_hat, y): # 本函数已保存在 d2lzh_pytorch包中方便以后使用
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并 没有除以 2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

def sgd(params, lr, batch_size): # 本函数已保存在 d2lzh_pytorch包中方便以后使用
    for param in params:
        param.data -= lr * param.grad / batch_size
# 注意这里更改param时用的param.data




