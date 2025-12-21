from torchvision import transforms
import torchvision
from torch.utils import data


data_loader_workers = 4
def load_data_fashion_mnist(batch_size,resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='../../data',train=True,download=True,transform=trans)
    mnist_test = torchvision.datasets.FashionMNIST(root='../../data',train=False,download=True,transform=trans)
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=data_loader_workers),
            data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=data_loader_workers))