import time
import argparse
from options.train_options import TrainOptions
from data.data_loader import DataLoader
from models.combogan_model import ComboGANModel
from util.visualizer import Visualizer
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-continue_train", "--continue_train", action='store_true', 
	help="Continue training: load the latest model")
ap.add_argument("-which_epoch", "--which_epoch", default=0, type=int, 
	help="Which epoch to load if continuing training")
ap.add_argument("-name", "--name", required=True, type=str, 
	help="Name of the experiment. It decides where to store samples and models")
ap.add_argument("-dataroot", "--dataroot", required=True, type=str,
	help="Path to images (should have subfolders trainA, trainB, valA, valB, etc)")
ap.add_argument("-n_domains", "--n_domains", required=True, type=int,
	help="Number of domains to transfer among")
ap.add_argument("-niter", "--niter", required=True, type=int,
	help="# of epochs at starting learning rate (try 50*n_domains)")
ap.add_argument("-niter_decay", "--niter_decay", required=True, type=int,
	help="# of epochs to linearly decay learning rate to zero (try 50*n_domains)")
ap.add_argument("-loadSize", "--loadSize", type=int, default=286,
	help="Scale images to this size")
ap.add_argument("-fineSize", "--fineSize", type=int, default=256,
	help="Then crop to this size")
ap.add_argument("-gpu_ids", "--gpu_ids", type=int, default=-1,
	help="GPU ids: 0, 1, or 2. use -1 for CPU")
args = vars(ap.parse_args())

    

def main():
    opt = TrainOptions().parse(args)

    dataset = DataLoader(opt)
    print('# training images = %d' % len(dataset))
    model = ComboGANModel(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    # Update initially if continuing
    if opt.which_epoch > 0:
        model.update_hyperparams(opt.which_epoch)

    prefix = os.path.join('.', 'checkpoints', opt.name, 'web')

    for epoch in range(opt.which_epoch + 1, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                visualizer.display_current_results(model.get_current_visuals(), epoch, prefix)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        model.update_hyperparams(epoch)

if __name__ == "__main__":
    main()