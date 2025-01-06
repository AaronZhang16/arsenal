import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

from model import MoCo_jpeg, Encoder, Encoder_simplified
from utility import checkpoint
from data import CustomDataset, jpeg_compression_collate_fn, random_jpeg_compression_collate_fn, \
    full_jpeg_compression_collate_fn, resize_collate_fn
import datetime

import matplotlib.pyplot as plt
import numpy as np
import cv2

import io
from PIL import Image
import random


class Trainer():
    def __init__(self, args, ckp):
        self.args = args
        self.writer = ckp.writer
        
        # 1.0 -----------------------------------------------------
        # set the training set
        # if self.args.jpeg_compress:
        #     self.dataset = CustomDataset(root_dir=self.args.dir_train)
        #     train_size = int(0.8 * len(self.dataset))
        #     val_size = len(self.dataset) - train_size
        #     train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])
        #     self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch, 
        #                               collate_fn=lambda x: full_jpeg_compression_collate_fn(x, batch_size=self.args.batch), 
        #                               shuffle=True, num_workers=4, drop_last=True)
        #     self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch, 
        #                         collate_fn=lambda x: full_jpeg_compression_collate_fn(x, batch_size=self.args.batch), 
        #                         shuffle=False, num_workers=4, drop_last=True)
        # else:
        #     self.dataset = CustomDataset(root_dir=self.args.dir_train)
        #     train_size = int(0.8 * len(self.dataset))
        #     val_size = len(self.dataset) - train_size
        #     train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])
        #     self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch, 
        #                               collate_fn=lambda x: moco_collate_fn(x), 
        #                               shuffle=True, num_workers=4, drop_last=True)
        #     self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch, 
        #                         collate_fn=lambda x: moco_collate_fn(x), 
        #                         shuffle=False, num_workers=4, drop_last=True)
        
        # 2.0 -------------------------------------------------------------------
        # if self.args.random_jpeg:
        #     self.dataset = CustomDataset(root_dir=self.args.dir_train)
        #     train_size = int(0.8 * len(self.dataset))
        #     val_size = len(self.dataset) - train_size
        #     train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])
        #     self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch, 
        #                                 collate_fn=lambda x: random_jpeg_compression_collate_fn(x), 
        #                                 shuffle=True, num_workers=4, drop_last=True)
        #     self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch, 
        #                         collate_fn=lambda x: random_jpeg_compression_collate_fn(x), 
        #                         shuffle=False, num_workers=4, drop_last=True)
        # else:
        #     self.dataset = CustomDataset(root_dir=self.args.dir_train)
        #     train_size = int(0.8 * len(self.dataset))
        #     val_size = len(self.dataset) - train_size
        #     train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])
        #     self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch, 
        #                                 collate_fn=lambda x: jpeg_compression_collate_fn(x), 
        #                                 shuffle=True, num_workers=4, drop_last=True)
        #     self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch, 
        #                         collate_fn=lambda x: jpeg_compression_collate_fn(x), 
        #                         shuffle=False, num_workers=4, drop_last=True)
        

        # 3.0 ------------------------------------------------------------------------
        if self.args.random_jpeg:
            self.dataset = CustomDataset(root_dir=self.args.dir_train)
            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])
            self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch, 
                                        collate_fn=lambda x: random_jpeg_compression_collate_fn(batch=x, fft_training=self.args.fft_learning), 
                                        shuffle=True, num_workers=4, drop_last=True)
        else:
            self.dataset = CustomDataset(root_dir=self.args.dir_train)
            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])
            # self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch, 
            #                             collate_fn=lambda x: jpeg_compression_collate_fn(x), 
            #                             shuffle=True, num_workers=4, drop_last=True)
            self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch, 
                                        collate_fn=lambda x: full_jpeg_compression_collate_fn(batch=x, fft_training=self.args.fft_learning), 
                                        shuffle=True, num_workers=4, drop_last=True)
        # self.val_loader = DataLoader(val_dataset, batch_size=4, 
        #                         collate_fn=lambda x: full_jpeg_compression_collate_fn(batch=x, batch_size=4), 
        #                         shuffle=True, num_workers=4, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=4, 
                                     collate_fn=lambda x: resize_collate_fn(x), 
                                     shuffle=True, num_workers=4, drop_last=True)
        
        # set the model 
        if self.args.model == 'DASR_Encoder':
            base_encoder = Encoder
        elif self.args.model == 'Simplified_DASR_Encoder':
            base_encoder = Encoder_simplified
        else:
            base_encoder = models.Resnet50
        self.model = MoCo_jpeg(base_encoder, dim=256).cuda()
        
        # set the optimizer
        if self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr = self.args.learning_rate, 
                                       momentum=0.9, weight_decay = self.args.weight_decay)
        elif self.args.optimizer == 'ADAM':
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.args.learning_rate, 
                                   weight_decay = self.args.weight_decay)

        # set the save path  
        self.ckp = ckp
        
    def train(self):
        self.model.train()        
        criterion = nn.CrossEntropyLoss()

        best_loss = float('inf')
        best_epoch = -1

        # weight_decay = self.args.weight_decay
        for epoch in range(self.args.epochs):
            total_loss = 0.0
            for i, images in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
                im_q = images[0].cuda()
                im_k = images[1].cuda()
                # self.ckp.write_log(f"     im_q's size: {im_q.size(0)}, {im_q.size(1)}, {im_q.size(2)}, {im_q.size(3)}; ")
                # self.ckp.write_log(f"     im_k's size: {im_k.size(0)}, {im_k.size(1)}, {im_k.size(2)}, {im_k.size(2)}; ")

                # ------------- 防止错误中断训练 -------------------------
                # if (im_q.size(0) != self.args.batch) | (im_k.size(0) != self.args.batch):
                #     print('Size error occurred in im_q') if im_q.size(0) != self.args.batch else print('Size error occurred in im_k')
                #     self.ckp.write_log(" !!!!!!!!!!!!! ERROR !!!!!!!!!!! ")
                #     self.ckp.write_log(f" Train error happened in Epoch: {epoch}, where im_q's size is {im_q.size(0)}, and im_k's size is {im_k.size(0)}")
                #     self.ckp.write_log(" ------------------------------- ")
                #     continue

                if self.args.jpeg_shuffle:

                    # 完全打乱---------------------------------------------------
                    # 获取打乱后的索引
                    indices = torch.randperm(im_k.size(0))
                    # 根据打乱后的索引重新排列 tensor
                    shuffled_im_k = im_k[indices]
                    # self.ckp.write_log(f"    shuffled_im_k's size: {shuffled_im_k.size(0)}, {shuffled_im_k.size(1)}, {shuffled_im_k.size(2)}, {shuffled_im_k.size(3)}")

                    # ------------- 防止错误中断训练 -------------------------
                    # if (shuffled_im_k.size(0) != self.args.batch):
                    #     print('Size error occurred in shuffled im_k')
                    #     self.ckp.write_log(" !!!!!!!!!!!!! ERROR !!!!!!!!!!! ")
                    #     self.ckp.write_log(f" Train error happened in Epoch: {epoch}, where shuffled_im_k's size is {shuffled_im_k.size(0)} ")
                    #     self.ckp.write_log(" ------------------------------- ")
                    #     continue
                    
                    logits, labels = self.model(im_q, shuffled_im_k)

                    # 部分打乱---------------------------------------------------
                    # shuffle_ratio = self.args.shuffle_rate
                    # # 获取打乱的数量
                    # shuffle_count = int(im_k.size(0) * shuffle_ratio)
                    # # 获取打乱部分的索引
                    # shuffled_indices = torch.randperm(shuffle_count)
                    # # 创建一个索引，包含打乱部分和未打乱部分
                    # indices = torch.cat((shuffled_indices, torch.arange(shuffle_count, im_k.size(0))))
                    # # 根据打乱后的索引重新排列 tensor
                    # shuffled_im_k = im_k[indices]
                    # # 继续执行后续操作
                    # logits, labels = self.model(im_q, shuffled_im_k)

                else:
                    logits, labels = self.model(im_q, im_k)
                
                loss = criterion(logits, labels)

                # L2正则化项
                # l2_reg = torch.tensor(0., requires_grad=True).cuda()
                # for param in self.model.parameters():
                #     l2_reg = l2_reg + torch.norm(param, 2)
                
                # 总损失 = 对比损失 + 正则化损失
                # loss = loss + weight_decay * l2_reg

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                total_loss += loss.item()

                if i % 100 == 0:  # 每100步记录一次
                    self.writer.add_scalar('Loss/Train', loss.item(), epoch * len(self.train_loader) + i)
            
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}")
            self.writer.add_scalar('Loss/Train_Avg', avg_loss, epoch)
        
            val_loss = self.validate(epoch)
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(self.ckp.dir_best, 'moco_best_checkpoint.pth'))

            self.ckp.write_log(f"Epoch: {epoch} | Train Loss: {avg_loss:.4f} | Valid Loss: {val_loss:.4f}")
            
        torch.save(self.model.state_dict(), os.path.join(self.ckp.dir_final, 'moco_final_checkpoint.pth'))
        print(f'Training Completed! The Final Loss: {val_loss}')
        print(f'The Best Val Loss: {best_loss}, which is in epoch {best_epoch}')
        self.ckp.write_log(f"Best Valid Loss: {best_loss}, in epoch {best_epoch}")

        self.writer.close()


    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for i, images in enumerate(self.val_loader):
                # im_q = images[0].cuda()
                # im_k = images[1].cuda()
                
                # # if self.args.jpeg_shuffle:
                # # 获取打乱后的索引
                # indices = torch.randperm(im_k.size(0))
                # # 根据打乱后的索引重新排列 tensor
                # shuffled_im_k = im_k[indices]
                # logits, labels = self.model(im_q, shuffled_im_k)
                # # else:
                # # logits, labels = self.model(im_q, im_k)
                    
                # loss = criterion(logits, labels)

                # total_loss += loss.item()
                for j in range(images.size(0)):
                    img1_pil = transforms.ToPILImage()(images[j])
                    img2_pil = transforms.ToPILImage()(images[(j+1) % images.size(0)])

                    img1_JPEG_1 = io.BytesIO()
                    img1_JPEG_2 = io.BytesIO()
                    img2_JPEG = io.BytesIO()

                    img1_pil.save(img1_JPEG_1, format='JPEG', quality=10)
                    img1_pil.save(img1_JPEG_2, format='JPEG', quality=90)
                    img2_pil.save(img2_JPEG, format='JPEG', quality=10)

                    img1_JPEG_1 = Image.open(img1_JPEG_1).convert('RGB')
                    img1_JPEG_2 = Image.open(img1_JPEG_2).convert('RGB')
                    img2_JPEG = Image.open(img2_JPEG).convert('RGB')

                    img1_JPEG_1_tensor = transforms.ToTensor()(img1_JPEG_1)
                    img1_JPEG_2_tensor = transforms.ToTensor()(img1_JPEG_2)
                    img2_JPEG_tensor = transforms.ToTensor()(img2_JPEG)

                    if self.args.fft_learning:
                        img1_JPEG_1_array = np.array(img1_JPEG_1)
                        img1_JPEG_2_array = np.array(img1_JPEG_2)
                        img2_JPEG_array = np.array(img2_JPEG)

                        img1_JPEG_1_fft = []
                        img1_JPEG_2_fft = []
                        img2_JPEG_fft = []

                        for i in range(img1_JPEG_1_array.shape[2]):
                            img1_JPEG_1_channel = img1_JPEG_1_array[:, :, i]  # 提取单个通道
                            img1_JPEG_2_channel = img1_JPEG_2_array[:, :, i]
                            img2_JPEG_channel = img2_JPEG_array[:, :, i]

                            img1_JPEG_1_fourier_transform = np.fft.fft2(img1_JPEG_1_channel)  # 进行二维傅里叶变换
                            img1_JPEG_2_fourier_transform = np.fft.fft2(img1_JPEG_2_channel)
                            img2_JPEG_fourier_transform = np.fft.fft2(img2_JPEG_channel)

                            img1_JPEG_1_magnitude_spectrum = np.log1p(np.abs(img1_JPEG_1_fourier_transform))
                            img1_JPEG_2_magnitude_spectrum = np.log1p(np.abs(img1_JPEG_2_fourier_transform))
                            img2_JPEG_magnitude_spectrum = np.log1p(np.abs(img2_JPEG_fourier_transform))

                            img1_JPEG_1_magnitude_spectrum_normalized = (img1_JPEG_1_magnitude_spectrum - img1_JPEG_1_magnitude_spectrum.min()) / (img1_JPEG_1_magnitude_spectrum.max() - img1_JPEG_1_magnitude_spectrum.min())
                            img1_JPEG_2_magnitude_spectrum_normalized = (img1_JPEG_2_magnitude_spectrum - img1_JPEG_2_magnitude_spectrum.min()) / (img1_JPEG_2_magnitude_spectrum.max() - img1_JPEG_2_magnitude_spectrum.min())
                            img2_JPEG_magnitude_spectrum_normalized = (img2_JPEG_magnitude_spectrum - img2_JPEG_magnitude_spectrum.min()) / (img2_JPEG_magnitude_spectrum.max() - img2_JPEG_magnitude_spectrum.min())

                            img1_JPEG_1_fft.append(img1_JPEG_1_magnitude_spectrum_normalized)
                            img1_JPEG_2_fft.append(img1_JPEG_2_magnitude_spectrum_normalized)
                            img2_JPEG_fft.append(img2_JPEG_magnitude_spectrum_normalized)

                        img1_JPEG_1_fft = np.transpose(np.stack(img1_JPEG_1_fft, axis=-1), (2, 0, 1))
                        img1_JPEG_2_fft = np.transpose(np.stack(img1_JPEG_2_fft, axis=-1), (2, 0, 1))
                        img2_JPEG_fft = np.transpose(np.stack(img2_JPEG_fft, axis=-1), (2, 0, 1))
                        
                        img1_JPEG_1_fft_tensor = torch.tensor(img1_JPEG_1_fft, dtype=torch.float32)
                        img1_JPEG_2_fft_tensor = torch.tensor(img1_JPEG_2_fft, dtype=torch.float32)
                        img2_JPEG_fft_tensor = torch.tensor(img2_JPEG_fft, dtype=torch.float32)

                        img1_JPEG_1 = torch.cat((img1_JPEG_1_tensor, img1_JPEG_1_fft_tensor), dim=1)
                        img1_JPEG_2 = torch.cat((img1_JPEG_2_tensor, img1_JPEG_2_fft_tensor), dim=1)
                        img2_JPEG = torch.cat((img2_JPEG_tensor, img2_JPEG_fft_tensor), dim=1)
                    else:
                        img1_JPEG_1 = img1_JPEG_1_tensor
                        img1_JPEG_2 = img1_JPEG_2_tensor
                        img2_JPEG = img2_JPEG_tensor
                    
                    with torch.no_grad():
                        img1_JPEG_1 = img1_JPEG_1.unsqueeze(0).cuda()
                        img1_JPEG_2 = img1_JPEG_2.unsqueeze(0).cuda()
                        img2_JPEG = img2_JPEG.unsqueeze(0).cuda()
                        img1_JPEG_1_repre = self.model(img1_JPEG_1)
                        img1_JPEG_2_repre = self.model(img1_JPEG_2)
                        img2_JPEG_repre = self.model(img2_JPEG)

                    repre_pos = torch.einsum('nc,nc->n', [img1_JPEG_1_repre, img2_JPEG_repre]).unsqueeze(-1)
                    repre_neg = torch.einsum('nc,nc->n', [img1_JPEG_1_repre, img1_JPEG_2_repre]).unsqueeze(-1)

                    val_logits = torch.cat([repre_pos, repre_neg], dim=1)

                    val_labels = torch.zeros(val_logits.shape[0], dtype=torch.long).cuda()

                    total_loss += criterion(val_logits, val_labels)

        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        self.writer.add_scalar('Loss/Validation', avg_loss, epoch)
        
        return avg_loss

