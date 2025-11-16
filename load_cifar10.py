from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch

class CIFAR10Torch(Dataset):
    def __init__(self, hf_split, transform=None):
        self.dataset = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        img = example["img"]          # PIL image
        label = example["label"]
        if self.transform:
            img = self.transform(img)
        return img, label


if __name__ == "__main__":
    hf_dataset = load_dataset("cifar10")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = CIFAR10Torch(hf_dataset["train"], transform=transform)
    test_dataset = CIFAR10Torch(hf_dataset["test"], transform=transform)

    # Set num_workers=0 for Windows (no multiprocessing)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    for images, labels in train_loader:
        print(images.shape, labels[:10])
        break
