import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
from PIL import Image
import io
import torch
import os
import logging
import numpy as np

def full_jpeg_compression_collate_fn(batch, fft_training=False, max_size=2000000):
    compressed_batch = []
    batch_size = len(batch)
    transform = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    
    quality = random.randint(10, 90)
    to_pil = transforms.ToPILImage()
    c, h, w = batch[0].shape
    while h * w * batch_size > max_size:
        h = int(h/2)
        w = int(w/2)
    
    for img in batch:
        img_pil = to_pil(img)
        img_pil = transforms.Resize((h, w))(img_pil)

        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer).convert('RGB')

        q_batch = transform(compressed_img)
        k_batch = transform(compressed_img)

        if fft_training:
            img_array = np.array(compressed_img)
            img_fft = []
            for i in range(img_array.shape[2]):
                channel = img_array[:, :, i]  # 提取单个通道
                fourier_transform = np.fft.fft2(channel)  # 进行二维傅里叶变换
                magnitude_spectrum = np.log1p(np.abs(fourier_transform))
                
                magnitude_normalized = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min()) # 归一化处理
                
                img_fft.append(magnitude_normalized)
            img_fft = np.transpose(np.stack(img_fft, axis=-1), (2, 0, 1))
            img_fft_tensor = torch.tensor(img_fft, dtype=torch.float32)

            q_batch = torch.cat((q_batch, img_fft_tensor), dim=1)
            k_batch = torch.cat((k_batch, img_fft_tensor), dim=1)
            
        compressed_batch.append([q_batch, k_batch])
    return torch.utils.data.dataloader.default_collate(compressed_batch)

def resize_collate_fn(batch, fft_training=False, max_size=2000000):
    resized_batch = []
    batch_size = len(batch)
    c, h, w = batch[0].shape

    while h * w * batch_size > max_size:
        h = int(h/2)
        w = int(w/2)
    
    for img in batch:
        resized_img = transforms.Resize((h, w))(img)

        resized_batch.append(resized_img)
    return torch.utils.data.dataloader.default_collate(resized_batch)

def moco_collate_fn(batch):
    resized_batch = []
    
    for img in batch:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        resized_batch.append([transform(img), transform(img)])

    return torch.utils.data.dataloader.default_collate(resized_batch)

# --------------------------------------------------------------------------------------------
def jpeg_compression_collate_fn(batch):
    compressed_batch = []
    transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    
    quality = random.randint(10, 90)
    to_pil = transforms.ToPILImage()
    
    for img in batch:
        img_pil = to_pil(img)
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer).convert('RGB')

        compressed_batch.append([transform(compressed_img), transform(compressed_img)])
    return torch.utils.data.dataloader.default_collate(compressed_batch)

def random_jpeg_compression_collate_fn(batch):
    compressed_batch = []
    transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    
    to_pil = transforms.ToPILImage()
    
    for img in batch:
        quality = random.randint(10, 90)
        img_pil = to_pil(img)
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer).convert('RGB')

        compressed_batch.append([transform(compressed_img), transform(compressed_img)])
    return torch.utils.data.dataloader.default_collate(compressed_batch)
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# def jpeg_compression_collate_fn(batch, max_size=10000000):
#     compressed_batch = []
#     transform = transforms.Compose([
#             # transforms.RandomResizedCrop(224),
#             # transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#             ])
    
#     quality = random.randint(1, 99)
#     to_pil = transforms.ToPILImage()

#     batch_size = len(batch)
#     c, h, w = batch[0].shape
#     while h * w * batch_size > max_size:
#         h = int(h/2)
#         w = int(w/2)

#     for img in batch:
#         img_pil = to_pil(img)
#         img_pil = transforms.Resize((h, w))(img_pil)
#         buffer = io.BytesIO()
#         img_pil.save(buffer, format='JPEG', quality=quality)
#         buffer.seek(0)
#         compressed_img = Image.open(buffer).convert('RGB')

#         compressed_batch.append([transform(compressed_img), transform(compressed_img)])
#     return torch.utils.data.dataloader.default_collate(compressed_batch)

# def random_jpeg_compression_collate_fn(batch, max_size=10000000):
#     compressed_batch = []
#     transform = transforms.Compose([
#             # transforms.RandomResizedCrop(224),
#             # transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#             ])
    
#     to_pil = transforms.ToPILImage()

#     batch_size = len(batch)
#     c, h, w = batch[0].shape
#     while h * w * batch_size > max_size:
#         h = int(h/2)
#         w = int(w/2)
    
#     for img in batch:
#         quality = random.randint(1, 99)
#         img_pil = to_pil(img)
#         img_pil = transforms.Resize((h, w))(img_pil)
#         buffer = io.BytesIO()
#         img_pil.save(buffer, format='JPEG', quality=quality)
#         buffer.seek(0)
#         compressed_img = Image.open(buffer).convert('RGB')

#         compressed_batch.append([transform(compressed_img), transform(compressed_img)])
#     return torch.utils.data.dataloader.default_collate(compressed_batch)
# --------------------------------------------------------------------------------------------

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]
        logging.basicConfig(filename='dataset_errors.log', level=logging.ERROR, 
                    format='%(asctime)s %(levelname)s %(message)s')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            img = Image.open(img_path).convert('RGB')
            img = transforms.ToTensor()(img)
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            logging.error(f"Error loading image {img_path}: {e}")
            return Image.new('RGB', (1920, 1080), color='white')
        

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]
        logging.basicConfig(filename='dataset_errors.log', level=logging.ERROR, 
                    format='%(asctime)s %(levelname)s %(message)s')

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        quality = random.randint(1, 99)

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer).convert('RGB')
        compressed_img = transforms.ToTensor()(compressed_img)

        return compressed_img, quality
    
class TestDataset_fft(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]
        logging.basicConfig(filename='dataset_errors.log', level=logging.ERROR, 
                    format='%(asctime)s %(levelname)s %(message)s')

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        quality = random.randint(1, 99)

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer).convert('RGB')

        img_array = np.array(compressed_img)
        img_fft = []
        for i in range(img_array.shape[2]):
            channel = img_array[:, :, i]  # 提取单个通道
            fourier_transform = np.fft.fft2(channel)  # 进行二维傅里叶变换
            magnitude_spectrum = np.log1p(np.abs(fourier_transform))
            magnitude_normalized = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())
    
            img_fft.append(magnitude_normalized)
        img_fft = np.transpose(np.stack(img_fft, axis=-1), (2, 0, 1))
        img_fft_tensor = torch.tensor(img_fft, dtype=torch.float32)
        
        compressed_img = transforms.ToTensor()(compressed_img)
        compressed_img = torch.cat((compressed_img, img_fft_tensor), dim=1)

        return compressed_img, quality
