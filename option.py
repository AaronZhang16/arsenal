import argparse

parser = argparse.ArgumentParser(description='Moco for making JPEG compression representation')

# Hardware Specifications
parser.add_argument('--n_threads', type=int, default=4,
                    help='number of threads for data loading')
parser.add_argument('--cpu', type=bool, default=False,
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=2,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data Specifications
parser.add_argument('--dir_train', type=str, default='/home/kaihang/dataset/DIV2K_train_HR',
                    help='training dataset directory')
parser.add_argument('--dir_demo', type=str, default='/home/kaihang/dataset/DIV2K_valid_HR',
                    help='demo image directory')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--dir_result', type=str, default='',
                    help='result parameter directory')
parser.add_argument('--dir_log', type=str, default='/home/kaihang/dataset/MocoJPEG/logs/',
                    help='tensorboard log directory')


# Training Specifications
parser.add_argument('--epochs', type=int, default=1600,
                    help='number of epochs to train the degradation encoder')
# parser.add_argument('--optimizer', default='SGD',
#                     choices=('SGD', 'ADAM', 'RMSprop'),
#                     help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--optimizer', default='SGD',
                    choices=('SGD', 'ADAM'),
                    help='optimizer to use (SGD | ADAM )')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--batch', type=int, default=64,
                    help='batch size')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--chop', action='store_true', default=False,
                    help='whether to chop the trainging image (when lack memory)')
parser.add_argument('--learning_rate', type=float, default=0.03,
                    help='learning rate of training')

# Proposed Methods Specifications
parser.add_argument('--model', choices=('Resnet50', 'DASR_Encoder', 'Simplified_DASR_Encoder'),
                    help='the base model for Moco')
parser.add_argument('--random_jpeg', action='store_true', default=False,
                    help='Whether to train the network with JPEG images in random compression rate')
parser.add_argument('--jpeg_shuffle', action='store_true', default=False,
                    help='Batch shuffling for content robust training')
parser.add_argument('--fft_learning', action='store_true', default=False,
                    help='FFT of the batch image')
parser.add_argument('--shuffle_rate', type=float, default=0.0,
                    help="the ratio of shuffle batch, from 0 to 1")
# conventional: random_jpeg: True, jpeg_shuffle: False
# proposed: random_jpeg: False, jpeg_shuffle: False / True

# Tester Specifications
parser.add_argument('--test', action='store_true', default=False,
                    help='Turn test mode')
parser.add_argument('--save_representation', action='store_true', default=False,
                    help='whether to save the degradation representation after testing')


args = parser.parse_args()

# args.scale = list(map(lambda x: float(x), args.scale.split('+')))