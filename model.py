import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MoCo_jpeg(nn.Module):
    def __init__(self, base_encoder, dim=256, K=65536, m=0.999, T=0.07):
        super(MoCo_jpeg, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # 创建两个编码器：一个用于query，一个用于key
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        
        # self.encoder_q = base_encoder()
        # self.encoder_k = base_encoder()

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # 初始化
            param_k.requires_grad = False  # key的编码器不需要梯度更新

        # 创建动量编码器队列
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k=None):
        # im_q与im_k是同一张图片的不同变换结果（正样本对）
        # 在test模式中，如果只是要生成representation，则im_k是不需要的

        # 获取打乱后的索引
        # indices = torch.randperm(im_k.size(0))
        # 根据打乱后的索引重新排列 tensor
        # shuffled_im_k = im_k[indices]

        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)
        
        if im_k is not None:
            with torch.no_grad():
                self._momentum_update_key_encoder()
                # -------------------------------------------
                k = self.encoder_k(im_k)
                # -------------------------------------------
                k = nn.functional.normalize(k, dim=1)

            # 计算对比损失
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # 正样本对的损失
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # 负样本对的损失

            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= self.T

            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            self._dequeue_and_enqueue(k)

            return logits, labels
        
        else:
            return q



class Encoder(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes # just for the API consistency with resnet50

        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return out
    

class Encoder_simplified(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes # just for the API consistency with resnet50

        super(Encoder_simplified, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return out