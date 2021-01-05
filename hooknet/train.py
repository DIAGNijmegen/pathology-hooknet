import os
import yaml

from argconfigparser import ArgumentConfigParser
from source.model import HookNet
from source.generator.batchgenerator import RandomBatchGenerator
from source.trainer import HookNetTrainer


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg

def train():
    """
    train function

    This train function is made for illustration and testing purposes only, as it uses a RandomBatchGenerator (i.e., random generated data)

    """

    # parse config and command line arguments
    parser = ArgumentConfigParser('./parameters.yml', description='HookNet')
    config = parser.parse_args()
    print(f'CONFIG: \n------\n{yaml.dump(config)}')

    # initialize model
    hooknet = HookNet(input_shape=config['input_shape'],
                      n_classes=config['n_classes'],
                      hook_indexes=config['hook_indexes'],
                      depth=config['depth'],
                      n_convs=config['n_convs'],
                      filter_size=config['filter_size'],
                      n_filters=config['n_filters'],
                      padding=config['padding'],
                      batch_norm=config['batch_norm'],
                      activation=config['activation'],
                      learning_rate=config['learning_rate'],
                      opt_name=config['opt_name'],
                      l2_lambda=config['l2_lambda'],
                      loss_weights=config['loss_weights'],
                      merge_type=config['merge_type'])

    # initialize batchgenerator
    batchgenerator = RandomBatchGenerator(batch_size=config['batch_size'],
                                          input_shape=hooknet.input_shape,
                                          output_shape=hooknet.output_shape,
                                          n_classes=config['n_classes'])

    # initialize trainer
    trainer = HookNetTrainer(model=hooknet,
                             batch_generator=batchgenerator,
                             epochs=config['epochs'],
                             steps=config['steps'],
                             batch_size=config['batch_size'],
                             output_path=config['output_path'])

    # train
    trainer.train()


if __name__ == "__main__":
    train()
