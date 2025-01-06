import os
import math
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as misc
import cv2
import torch

from torch.utils.tensorboard import SummaryWriter

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        
        if args.dir_result == '':
            self.dir = 'experiments/' + str(now)
        else:
            self.dir = 'experiments/' + args.dir_result

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)
            
        self.dir_log = self.dir + '/logs'
        self.dir_best = self.dir + '/best'
        self.dir_final = self.dir + '/final'

        _make_dir(self.dir)
        _make_dir(self.dir_best)
        _make_dir(self.dir_final)
        _make_dir(self.dir_log)

        open_type = 'a' if os.path.exists(self.dir + '/config.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.writer = SummaryWriter(log_dir=self.dir + '/logs')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        # self.plot_psnr(epoch)
        # torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        # torch.save(
        #     trainer.optimizer.state_dict(),
        #     os.path.join(self.dir, 'optimizer.pt')
        # )
    
    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        # print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    # def plot_psnr(self, epoch):
    #     axis = np.linspace(1, epoch, epoch)
    #     label = 'SR on {}'.format(self.args.data_test)
    #     fig = plt.figure()
    #     plt.title(label)
    #     for idx_scale, scale in enumerate(self.args.scale):
    #         plt.plot(
    #             axis,
    #             self.log[:, idx_scale].numpy(),
    #             label='Scale {}'.format(scale)
    #         )
    #     plt.legend()
    #     plt.xlabel('Epochs')
    #     plt.ylabel('PSNR')
    #     plt.grid(True)
    #     plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
    #     plt.close(fig)

    def save_results(self, filename, save_list, scale):
        filename = '{}/results/{}_x{}_'.format(self.dir, filename, scale)

        normalized = save_list[0][0].data.mul(255 / self.args.rgb_range)
        ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
        misc.imsave('{}{}.png'.format(filename, 'SR'), ndarr)




        