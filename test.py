import time
import os
import argparse
from options.test_options import TestOptions
from data.data_loader import DataLoader
from models.combogan_model import ComboGANModel
from util.visualizer import Visualizer
from util import html

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-name", "--name", required=True, type=str, 
	help="Name of the experiment. It decides where to store samples and models")
ap.add_argument("-dataroot", "--dataroot", required=True, type=str,
	help="Path to images (should have subfolders test0, test1)")
ap.add_argument("-phase", "--phase", type=str, default='test',
	help="Train, val, test, etc (determines name of folder to load from)")
ap.add_argument("-which_epoch", "--which_epoch", required=True, type=int,
	help="Which epoch to load for inference?")
ap.add_argument("-serial_test", "--serial_test", action='store_true',
	help="Read each image once from folders in sequential order")
ap.add_argument("-n_domains", "--n_domains", required=True, type=int,
	help="Number of domains to transfer among")
ap.add_argument("-loadSize", "--loadSize", type=int, default=286,
	help="Scale images to this size")
ap.add_argument("-gpu_ids", "--gpu_ids", type=int, default=-1,
	help="GPU ids: 0, 1, or 2. use -1 for CPU")
args = vars(ap.parse_args())

def main():
    opt = TestOptions().parse(args)
    
    dataset = DataLoader(opt)
    model = ComboGANModel(opt)
    
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%d' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %d' % (opt.name, opt.phase, opt.which_epoch))
    # store images for matrix visualization
    vis_buffer = []
    prefix = os.path.join('.', 'results', opt.name, f'test_{opt.which_epoch}')

    # test
    for i, data in enumerate(dataset):
        # if not opt.serial_test and i >= opt.how_many:
            # break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals(testing=True)
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path, prefix)

        if opt.show_matrix:
            vis_buffer.append(visuals)
            if (i+1) % opt.n_domains == 0:
                save_path = os.path.join(web_dir, 'mat_%d.png' % (i//opt.n_domains))
                visualizer.save_image_matrix(vis_buffer, save_path)
                vis_buffer.clear()

    webpage.save()

if __name__ == "__main__":
    main()
