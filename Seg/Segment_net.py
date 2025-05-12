from . import archs
import argparse
from .utils import AverageMeter, str2bool
import torch
ARCH_NAMES = archs.__all__
def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=250, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=250, type=int,
                        help='image height')
    
    parser.add_argument('--num_workers', default=0, type=int)

    config = parser.parse_args()

    return config


def get_segment_model(if_load):
    config=vars(parse_args())
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    if if_load == True:
        state_dict = torch.load('/media/hp/新加卷/XNW/Hiearchical_RL/Seg/Segment_model.pth')
        model.load_state_dict(state_dict)
    return model