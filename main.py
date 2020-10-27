import os
import sys
import argparse
import logging
import torch

from datasets.data_loader import get_dataset_loader
from solver import Solver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    logger.info('\t Called with args:')
    logger.info(args)

    if not torch.cuda.is_available():
        sys.exit('Need a CUDA device to run the code.')
    else:
        args.cuda = True

    save_dir = os.path.join(args.save_dir, args.model, args.pattern, args.version)
    # create directories if not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger.info('\t output will be saved to {}'.format(save_dir))

    log_dir = os.path.join(save_dir, args.log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger.info('\t logs will be saved to {}'.format(log_dir))

    args.save_dir = save_dir
    args.log_dir = log_dir
    if args.pattern == 'dct':
        data_root = args.data_root+'_dct'
    if args.pattern == 'fft':
        data_root = args.data_root+'_fft'

    else:
        data_root = args.data_root


    if args.mode == 'train':
        dataloader = get_dataset_loader(data_root, args.training_dataset, args.batch_size, args.num_workers, args.mode)
        val_dataloader = get_dataset_loader(data_root, args.val_dataset, args.batch_size, args.num_workers, 'val')
        solver = Solver(args, dataloader, val_dataloader)
        solver.train()
    if args.mode == 'test':
        test_dataloader = get_dataset_loader(data_root, args.test_dataset, args.batch_size, args.num_workers, 'test')
        solver = Solver(args, None, test_dataloader)
        solver.val()
        #solver.test()
    elif args.mode == 'val':
        #dataloader = get_dataset_loader(data_root, args.training_dataset, args.batch_size, args.num_workers, args.mode)
        val_dataloader = get_dataset_loader(data_root, args.val_dataset, args.batch_size, args.num_workers, 'val')
        solver = Solver(args, None, val_dataloader)
        ##solver.val()
        solver.test()


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a classifier network')

    # Model hyper-parameters
    #parser.add_argument('--data_root', default='/data/data/DGM_data')
    parser.add_argument('--data_root', default='../Data_Frequency')

    parser.add_argument('--training_dataset', dest='training_dataset',
                        help='training dataset',
                        default=['ffhq', 'stylegan'], type=list, choices=['ffhq', 'stylegan'])
    parser.add_argument('--val_dataset', dest='val_dataset',
                        help='validate dataset',
                        default=['ffhq', 'stylegan'], type=list,
                        choices=['AttGAN_256', 'ffhq', 'stylegan', 'stargan_v2_256','celebA'])
    parser.add_argument('--test_dataset', dest='test_dataset',
                        help='test dataset',
                        default=[ 'stargan_v2_256'], type=list,
                        choices=['AttGAN_256', 'stargan_v2_256','celebA'])
    parser.add_argument('--model', type=str, default='vgg', choices=['resnet50', 'vgg', 'densenet', 'D'])
    parser.add_argument('--pattern','-p',type=str, default='fft', choices=['raw', 'dct','fft'])
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'val'])

    # Path
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="./output",
                        type=str)
    parser.add_argument('--log_dir', dest='log_dir', help="directory to log",
                        default='log', type=str)

    # Training settings
    parser.add_argument('--num_workers', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--batch_size', dest='batch_size',
                        help='batch_size',
                        default=32, type=int)
    parser.add_argument('--max_iter', dest='max_iter',
                        help='max_iter',
                        default=50000, type=int)
    parser.add_argument('--lr', dest='lr', help='starting learning rate', type=float, default=0.00005)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio', type=float, default=0.9)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='learning rate decay step', type=int, default=1000)
    parser.add_argument("--version", type=str, default='lr5e-5_step1000_ratio0.9')
    parser.add_argument('--o', dest='optimizer', help='Training optimizer.', default='Adam')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--use_srm', dest='use_srm', type=bool,
                        help='whether use srm filter', default=False)
    # Step size
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--save_interval', dest='save_interval',
                        help='number of iterations to save',
                        default=2000, type=int)
    parser.add_argument('--resume_iter', type=int, default=None, help='resume training from this step')
    # Misc6
    parser.add_argument('--seed', type=int, default=22, help='random seed (default: 1)')
    parser.add_argument('--use_tensorboard', dest='use_tensorboard',
                        help='whether use tensorflow tensorboard',
                        default=True, type=bool)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default='model_10000.pth', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    main()
