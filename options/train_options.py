from random import choices
from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=1000, help='frequency of saving sample output (epochs).')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console (iterations)')
        parser.add_argument('--tb_runs_dir', type=str, default='/home/ubuntu/runs/', help='where the tensorboard run logs are saved.')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        # specialized loss parameters
        parser.add_argument('--bg_dc', type=str, default='none', help='type of background data-consistency. [loss | encoding | loss_and_encoding | none]')
        parser.add_argument('--bg_dc_kernel_size', type=int, default=5, help='size of the gaussian kernel used to blur the synthetic pedestrian mask to simulate decaying data consistency.')
        parser.add_argument('--bg_dc_sigma', type=float, default=2.0, help='standard deviation used to create gaussian kernel used to blur the synthetic pedestrian mask to simulate decaying data consistency.')
        parser.add_argument('--bg_dc_expand_radius', type=int, default=3, help='how many pixels to expand the seg masks by before applying the gaussian filter used to blur the synth ped mask to simulate decaying data consistency.')
        parser.add_argument('--bg_dc_loss_type', type=str, choices=['L1', 'L2'], default='L2', help='which loss to use for the background data consistency.')
        parser.add_argument('--bg_dc_loss_weight', type=float, default=1, help='scaling factor to multiply background data consistency loss by.')

        self.isTrain = True
        return parser
