import os
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import datetime
import tqdm
import pandas as pd

from model import MoCo_jpeg, Encoder, Encoder_simplified
from data import TestDataset, TestDataset_fft


class Tester():
    def __init__(self, args):
        self.args = args

        # set the model 
        if self.args.model == 'DASR_Encoder':
            base_encoder = Encoder
        elif self.args.model == 'Simplified_DASR_Encoder':
            base_encoder = Encoder_simplified
        else:
            base_encoder = models.Resnet50
        self.model = MoCo_jpeg(base_encoder, dim=256).cuda()
        self.model.load_state_dict(torch.load('experiments/' + self.args.dir_result + '/best/moco_best_checkpoint.pth'))
        
        if self.args.fft_learning:
            self.test_dataset = TestDataset_fft(root_dir=self.args.dir_demo)
        else:
            self.test_dataset = TestDataset(root_dir=self.args.dir_demo)
            
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0)

    def generate_representation(self):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.model.eval()

        excel_path = os.path.join('demo', self.args.dir_result + '_' + now + '_representations.xlsx')
        
        with torch.no_grad():
            all_representation = []
            for i in range(10):
                for image, quality in self.test_loader:
                    im_q = image.cuda()
                    
                    representation = self.model(im_q)
                    
                    representation = representation.cpu().numpy()

                    all_representation.append(np.append(representation, quality))
            
            df = pd.DataFrame(all_representation)
            df.to_excel(excel_path, index=False)






        