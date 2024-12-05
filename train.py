import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from torch.amp import GradScaler, autocast  # Updated import for GradScaler and autocast

scaler = GradScaler()  # Initialize GradScaler for mixed precision training

if __name__ == '__main__':
    opt = TrainOptions().parse()   # Get training options
    dataset = create_dataset(opt)  # Create dataset
    dataset_size = len(dataset)    # Get the number of images in the dataset
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # Create model
    model.setup(opt)               # Regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # Create visualizer for displaying/saving results
    total_iters = 0                # The total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):  # Outer loop for epochs
        epoch_start_time = time.time()  # Timer for entire epoch
        epoch_iter = 0                  # Reset per-epoch iteration count
        visualizer.reset()              # Reset visualizer for saving results
        model.update_learning_rate()    # Update learning rates at the beginning of the epoch

        for i, data in enumerate(dataset):  # Inner loop for each batch
            iter_start_time = time.time()   # Timer for iteration
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)           # Unpack and preprocess input data
            with autocast(device_type="cuda"):  # Updated usage of autocast
                model.optimize_parameters()  # Optimize network parameters

            if total_iters % opt.display_freq == 0:  # Display and save results
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:  # Print and log training losses
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, 0)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:  # Save the latest model checkpoint
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

        if epoch % opt.save_epoch_freq == 0:  # Save model checkpoint at epoch intervals
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
