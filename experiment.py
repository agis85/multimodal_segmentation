
import argparse
import importlib
import json
import logging
import os

import git
import matplotlib
import numpy
import comet_ml
matplotlib.use('Agg')  # environment for non-interactive environments

from easydict import EasyDict


class Experiment(object):
    def __init__(self):
        self.log = None

    def init_logging(self, config):
        if not os.path.exists(config.folder):
            os.makedirs(config.folder)
        logging.basicConfig(filename=config.folder + '/logfile.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
        logging.getLogger().addHandler(logging.StreamHandler())

        self.log = logging.getLogger()
        self.log.debug(config.items())
        self.log.info('---- Setting up experiment at ' + config.folder + '----')

    def get_config(self, split, args):
        """
        Read a config file and convert it into an object for easy processing.
        :param split: the cross-validation split id
        :param args:  the command arguments
        :return: config object(namespace)
        """
        config_script = args.config

        config_dict = importlib.import_module('configuration.' + config_script).get()
        config = EasyDict(config_dict)
        config.split = split

        if (hasattr(config, 'randomise') and config.randomise) or (hasattr(args, 'randomise') and args.randomise):
            config.randomise   = True
            config.folder     += '_randomise'

        config.n_pairs = 1
        if (hasattr(config, 'automatedpairing') and config.automatedpairing) or \
                (hasattr(args, 'automatedpairing') and args.automatedpairing):
            config.automatedpairing = True
            config.folder     += '_automatedpairing'
            config.n_pairs = 3

        l_mix = config.l_mix
        if hasattr(args, 'l_mix'):
            config.l_mix = float(args.l_mix)
            l_mix = args.l_mix
        config.folder += '_l%s' % l_mix

        config.folder += '_' + str(config.modality)
        config.folder += '_split%s' % split
        config.folder = config.folder.replace('.', '')

        if args.test_dataset:
            print('Overriding default test dataset')
            config.test_dataset = args.test_dataset

        config.githash = git.Repo(search_parent_directories=True).head.object.hexsha

        self.save_config(config)
        return config

    def save_config(self, config):
        if not os.path.exists(config.folder):
            os.makedirs(config.folder)
        with open(config.folder + '/experiment_configuration.json', 'w') as outfile:
            json.dump(dict(config.items()), outfile)

    def run(self):
        args = Experiment.read_console_parameters()
        configuration = self.get_config(int(args.split), args)
        self.init_logging(configuration)
        self.run_experiment(configuration, args.test)

    def run_experiment(self, configuration, test):
        executor = self.get_executor(configuration, test)

        if test:
            executor.test()
        else:
            executor.train()
            def default(o):
                if isinstance(o, numpy.int64): return int(o)
                raise TypeError
            with open(configuration.folder + '/experiment_configuration.json', 'w') as outfile:
                json.dump(vars(configuration), outfile, default=default)
            executor.test()

    @staticmethod
    def read_console_parameters():
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--config', default='', help='The experiment configuration file', required=True)
        parser.add_argument('--test', help='Evaluate the model on test data', type=bool)
        parser.add_argument('--test_dataset', help='Override default test dataset', choices=['chaos'])
        parser.add_argument('--split', help='Data split to run.', required=True)
        parser.add_argument('--l_mix', help='Percentage of labelled data')
        parser.add_argument('--automatedpairing', help='Use weighted cost for training', type=bool)
        parser.add_argument('--randomise', help='Randomise multimodal pairs', type=bool)

        return parser.parse_args()

    def get_executor(self, config, test):
        # Initialise model
        module_name = config.model.split('.')[0]
        model_name = config.model.split('.')[1]
        model = getattr(importlib.import_module('models.' + module_name), model_name)(config)
        model.build()

        # Initialise executor
        module_name = config.executor.split('.')[0]
        model_name = config.executor.split('.')[1]
        executor = getattr(importlib.import_module('model_executors.' + module_name), model_name)(config, model)
        return executor


if __name__ == '__main__':
    exp = Experiment()
    exp.run()
