from torchvision import transforms

def default_transform(resolution:int):
    return transforms.Compose([transforms.Resize((resolution,resolution)), transforms.ToTensor()])
