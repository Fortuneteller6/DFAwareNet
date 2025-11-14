from model.utils import *

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='WaveletMamba_For_SIRST')
    # choose model
    parser.add_argument('--model', type=str, default='WaveletMamba',
                        help='model name: WaveletMamba')

    # parameter for WaveletMamba
    parser.add_argument('--deep_supervision', type=str, default='True', help='True or False (model==WaveletMamba)')


    # data and pre-process
    parser.add_argument('--dataset', type=str, default='IRSTD-1K', help='dataset name: IRSTD-1K, NUAA-SIRST, NUDT-SIRST')
    parser.add_argument('--st_model', type=str, default='IRSTD-1K_WaveletGuidedMamba_03_08_2024_18_52_24_wDS')
    parser.add_argument('--model_dir', type=str, default = 'IRSTD-1K_WaveletGuidedMamba_03_08_2024_18_52_24_wDS/mIoU__WaveletGuidedMamba_IRSTD-1K_epoch.pth')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--test_size', type=float, default='0.5', help='when --mode==Ratio')
    parser.add_argument('--root', type=str, default='dataset/')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--split_method', type=str, default='80_20', help='80_20 (for three datasets)')
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3, help='in_channel=3 for pre-process')
    parser.add_argument('--base_size', type=int, default=256, help='base image size')
    parser.add_argument('--crop_size', type=int, default=256, help='crop image size')

    #  hyper params for training
    parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N', help='input batch size for testing (default: 32)')

    # cuda and logging
    parser.add_argument('--gpus', type=str, default='0', help='Training with GPUs, you can specify 1,3 for example.')

    # ROC threshold
    parser.add_argument('--ROC_thr', type=int, default=10, help='crop image size')

    args = parser.parse_args()

    # the parser
    return args