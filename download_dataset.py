from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset = datasets.CIFAR10(root='../dataset', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='../dataset', train=False, download=True, transform=transform)