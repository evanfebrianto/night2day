import os
from util import util
import torch
import easydict

class BaseTestOptions():
    def __init__(self):
        self.parser = None
        self.initialized = False
        
    def initialize(self):
        self.parser = easydict.EasyDict({
            "name" : 'robotcar_2day',
            "checkpoints_dir" : './checkpoints',
            "dataroot" : './datasets/test_dataset/',
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
            "nThreads" : 1,
            "display_id" : 0,
            "display_port" : 8097,
            "display_winsize" : 256,
            "display_single_pane_ncols" : 0,
            "results_dir" : './results/',
            "aspect_ratio" : 1.0,
            "which_epoch" : 150,
            "phase" : 'test',
            "how_many" : 50,
            "serial_test" : True,
            "autoencode" : False,
            "reconstruct" : False,
            "show_matrix" : False
        })
        self.initialized = True
        
    def parse(self, args=None):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser #.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        if args is not None:
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

class TestOptions(BaseTestOptions):
    def initialize(self):
        BaseTestOptions.initialize(self)
        self.isTrain = False


# from .base_options import BaseOptions


# class TestOptions(BaseOptions):
#     def initialize(self):
#         BaseOptions.initialize(self)
#         self.isTrain = False

#         self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
#         self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')

#         self.parser.add_argument('--which_epoch', required=True, type=int, help='which epoch to load for inference?')
#         self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc (determines name of folder to load from)')

#         self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run (if serial_test not enabled)')
#         self.parser.add_argument('--serial_test', action='store_true', help='read each image once from folders in sequential order')

#         self.parser.add_argument('--autoencode', action='store_true', help='translate images back into its own domain')
#         self.parser.add_argument('--reconstruct', action='store_true', help='do reconstructions of images during testing')

#         self.parser.add_argument('--show_matrix', action='store_true', help='visualize images in a matrix format as well')
