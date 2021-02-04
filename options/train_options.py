import os
from util import util
import torch
import easydict

class BaseTrainOptions():
    def __init__(self):
        self.parser = None
        self.initialized = False
        
    def initialize(self):
        self.parser = easydict.EasyDict({
            "name" : 'robotcar_night2day',
            "checkpoints_dir" : './checkpoints',
            "dataroot" : './datasets/train_dataset/',
            "n_domains" : 2,
            "max_dataset_size" : float("inf"),
            "resize_or_crop" : 'resize_and_crop',
            "no_flip" : False,
            "loadSize" : 512,
            "fineSize" : 256,
            "batchSize" : 1,
            "input_nc" : 3,
            "output_nc" : 3,
            "ngf" : 64,
            "ndf" : 64,
            "netG_n_blocks" : 9,
            "netG_n_shared" : 0,
            "netD_n_layers" : 4,
            "norm" : 'instance',
            "use_dropout" : False,
            "gpu_ids" : -1,
            "nThreads" : 2,
            "display_id" : 0,
            "display_port" : 8097,
            "display_winsize" : 256,
            "display_single_pane_ncols" : 0,
            "continue_train": False,
            "which_epoch": 0,
            "phase": 'train',
            "niter": 75,
            "niter_decay": 75,
            "lr": 0.0002,
            "beta1": 0.5,
            "lambda_cycle": 10.0,
            "lambda_identity": 0.0,
            "lambda_latent": 0.0,
            "lambda_forward": 0.0,
            "save_epoch_freq": 5,
            "display_freq": 100,
            "print_freq": 100,
            "pool_size": 50,
            "no_html": False
        })
        self.initialized = True
        
    def parse(self, args):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser#.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        for i in args:
            self.opt[i] = args[i]
    
        str_ids = [self.opt.gpu_ids] #.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt

class TrainOptions(BaseTrainOptions):
    def initialize(self):
        BaseTrainOptions.initialize(self)
        self.isTrain = True

# from .base_options import BaseOptions


# class TrainOptions(BaseOptions):
#     def initialize(self):
#         BaseOptions.initialize(self)
#         self.isTrain = True

#         self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
#         self.parser.add_argument('--which_epoch', type=int, default=0, help='which epoch to load if continuing training')
#         self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc (determines name of folder to load from)')

#         self.parser.add_argument('--niter', required=True, type=int, help='# of epochs at starting learning rate (try 50*n_domains)')
#         self.parser.add_argument('--niter_decay', required=True, type=int, help='# of epochs to linearly decay learning rate to zero (try 50*n_domains)')

#         self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for ADAM')
#         self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of ADAM')

#         self.parser.add_argument('--lambda_cycle', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
#         self.parser.add_argument('--lambda_identity', type=float, default=0.0, help='weight for identity "autoencode" mapping (A -> A)')
#         self.parser.add_argument('--lambda_latent', type=float, default=0.0, help='weight for latent-space loss (A -> z -> B -> z)')
#         self.parser.add_argument('--lambda_forward', type=float, default=0.0, help='weight for forward loss (A -> B; try 0.2)')

#         self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
#         self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
#         self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')

#         self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
#         self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
