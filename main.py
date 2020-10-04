import argparse
import json
import pathlib
import os

from trainer.lsr_trainer import LSRTrainer
from trainer.kd_trainer import KDTrainer
from trainer.pruning_trainer import PruningTrainer
from trainer.move_trainer import MOVETrainer


def main(run_type, cfg, experiment_name):
    if run_type == 'train':
        if cfg['exp_type'] == 'lsr':
            trainer = LSRTrainer(cfg, experiment_name)
        elif cfg['exp_type'] == 'kd':
            trainer = KDTrainer(cfg, experiment_name)
        elif cfg['exp_type'] == 'pruning':
            trainer = PruningTrainer(cfg, experiment_name)
        elif cfg['exp_type'] == 'baseline':
            trainer = MOVETrainer(cfg, experiment_name)
        else:
            Exception('Training mode not understood.')
        trainer.train()
    elif run_type == 'test':
        pass
    else:
        print('Run type not understood.')


if __name__ == '__main__':
    path = pathlib.Path(__file__).parent.absolute()

    parser = argparse.ArgumentParser(description='Training and evaluation code for Re-MOVE experiments.')
    parser.add_argument('-rt',
                        '--run_type',
                        type=str,
                        default='train',
                        choices=['train', 'test'],
                        help='Whether to run the training or the evaluation script.')
    parser.add_argument('-mp',
                        '--main_path',
                        type=str,
                        default='{}'.format(path),
                        help='Path to the working directory.')
    parser.add_argument('--exp_type',
                        type=str,
                        default='lsr',
                        choices=['lsr', 'kd', 'pruning', 'baseline'],
                        help='Type of experiment to run.')
    parser.add_argument('-pri',
                        '--pruning_iterations',
                        type=int,
                        default=None,
                        help='Number of iterations for pruning.')
    parser.add_argument('-tf',
                        '--train_file',
                        type=str,
                        default=None,
                        help='Path for training file. If more than one file are used, '
                             'write only the common part.')
    parser.add_argument('-ch',
                        '--chunks',
                        type=int,
                        default=None,
                        help='Number of chunks for training set.')
    parser.add_argument('-vf',
                        '--val_file',
                        type=str,
                        default=None,
                        help='Path for validation data.')
    parser.add_argument('-noe',
                        '--num_of_epochs',
                        type=int,
                        default=None,
                        help='Number of epochs for training.')
    parser.add_argument('-emb',
                        '--emb_size',
                        type=int,
                        default=None,
                        help='Size of the final embeddings.')
    parser.add_argument('-bs',
                        '--batch_size',
                        type=int,
                        default=None,
                        help='Batch size for training iterations.')
    parser.add_argument('-l',
                        '--loss',
                        type=int,
                        default=None,
                        choices=[0, 1, 2, 3],
                        help='Which loss to use for training. 0 for Triplet, '
                             '1 for ProxyNCA, 2 for NormalizedSoftmax, and 3 for Group loss.')
    parser.add_argument('-kdl',
                        '--kd_loss',
                        type=str,
                        default=None,
                        choices=['distance', 'cluster'],
                        help='Which loss to use for Knowledge Distillation training.')
    parser.add_argument('-ms',
                        '--mining_strategy',
                        type=int,
                        default=None,
                        choices=[0, 1, 2, 3],
                        help='Mining strategy for Triplet loss. 0 for random, 1 for semi-hard, '
                             '2 for hard, 3 for semi-hard with using all positives.')
    parser.add_argument('-ma',
                        '--margin',
                        type=float,
                        default=None,
                        help='Margin for Triplet loss.')
    parser.add_argument('-o',
                        '--optimizer',
                        type=int,
                        default=None,
                        choices=[0, 1],
                        help='Optimizer for training. 0 for SGD, 1 for Ranger.')
    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        default=None,
                        help='Base learning rate for the optimizer.')
    parser.add_argument('-flr',
                        '--finetune_learning_rate',
                        type=float,
                        default=None,
                        help='Learning rate for finetuning the feature extractor for LSR training.')
    parser.add_argument('-mo',
                        '--momentum',
                        type=float,
                        default=None,
                        help='Value for momentum parameter for SGD.')
    parser.add_argument('-lrs',
                        '--lr_schedule',
                        type=int,
                        nargs='+',
                        default=None,
                        help='Epochs for reducing the learning rate. Multiple arguments are appended in a list.')
    parser.add_argument('-lrg',
                        '--lr_gamma',
                        type=float,
                        default=None,
                        help='Step size for learning rate scheduler.')
    parser.add_argument('-pl',
                        '--patch_len',
                        type=int,
                        default=None,
                        help='Number of frames for each input in time dimension (only for training).')
    parser.add_argument('-da',
                        '--data_aug',
                        type=int,
                        default=None,
                        choices=[0, 1],
                        help='Whether to use data augmentation while training.')
    parser.add_argument('-nw',
                        '--num_workers',
                        default=None,
                        type=int,
                        help='Num of workers for the data loader.')
    parser.add_argument('-ofb',
                        '--overfit_batch',
                        type=int,
                        default=None,
                        help='Whether to overfit a single batch. It may help with revealing problems with training.')

    args = parser.parse_args()

    experiment_name = 're-move_{}'.format(args.exp_type)

    with open(os.path.join(path, 'data/{}_defaults.json'.format(args.exp_type))) as f:
        cfg = json.load(f)

    for key in args.__dict__.keys():
        if getattr(args, key) is None:
            setattr(args, key, cfg[key])

    for key in cfg.keys():
        if key == 'abbr':
            pass
        else:
            if cfg[key] != getattr(args, key):
                val = '{}'.format(getattr(args, key))
                val = val.replace('.', '-')
                experiment_name = '{}_{}_{}'.format(experiment_name, cfg['abbr'][key], val)

    main(args.run_type, args.__dict__, experiment_name)
