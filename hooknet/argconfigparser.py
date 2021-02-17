from typing import Dict
import yaml
import argparse
import sys
from pprint import pprint
import ast
import os.path

from argparse import ArgumentParser


class RecursiveLoader(yaml.Loader):

    """
    Yaml loader for including yaml files in yaml file with !include or !import

    """

    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super().__init__(stream)
        RecursiveLoader.add_constructor("!include", RecursiveLoader.include)
        RecursiveLoader.add_constructor("!import", RecursiveLoader.include)

    def include(self, node):
        if isinstance(node, yaml.ScalarNode):
            return self.extractFile(self.construct_scalar(node))

        elif isinstance(node, yaml.SequenceNode):
            result = []
            for filename in self.construct_sequence(node):
                result += self.extractFile(filename)
            return result

        elif isinstance(node, yaml.MappingNode):
            result = {}
            for k, v in self.construct_mapping(node).iteritems():
                result[k] = self.extractFile(v)
            return result

        else:
            print("Error:: unrecognised node type in !include statement")
            raise yaml.constructor.ConstructorError

    def extractFile(self, filename):
        filepath = os.path.join(self._root, filename)
        with open(filepath, "r") as f:
            return yaml.load(f, RecursiveLoader)


def _str2value(v):
    if v.lower() in ("yes", "true", "t", "y"):
        return True
    elif v.lower() in ("no", "false", "f", "n"):
        return False
    elif v.lower() in ["none", "null"]:
        return None
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class ArgumentConfigParser(ArgumentParser):
    """
    This class parses a given config yaml file and overwrites same named parameters with command line arguments
    Lists are set by spaces between values (e.g., --list_argument list_item1 list_item2 list_item3).
    Dictionaries in dictionaries are set by ':' signs between values (e.g., --parent_dict_key:child_dict_key child_dict_value).
    """

    def __init__(
        self, config_file_path: str, notebook: bool = False, description: str = ""
    ) -> None:
        """
        Parameters
        ----------
        config_file_path : str
            path to the config file
        notebook : bool
            if notebook is True no command line arguments are parsed (useful when working in a notebook)

        Returns
        -------
        config: Dict
            All key-value pairs defined in the config file, possibly overwritten by command line arguments

        """

        self._config_file_path = config_file_path

        self._config = None

        with open(self._config_file_path) as json_config:
            self._config = yaml.load(json_config, Loader=RecursiveLoader)

        if notebook:
            return self._config

        super().__init__(description=description)
        self.add_argument("-c", "--config", help="config file location", required=False)
        self._add_arguments(self._config)

    def parse_args(self) -> Dict:
        args = vars(super().parse_args())

        # if config file is given via command line, set config file
        if args["config"]:
            with open(args["config"]) as yml_config:
                self._config = yaml.load(yml_config, Loader=RecursiveLoader)
        else:
            args["config"] = self._config_file_path

        return self._set_arguments(self._config, args)

    def _set_arguments(self, config, args, rekey=None) -> Dict:
        # if value is not set through arguments set value
        for key, value in config.items():
            value_type = type(value)
            setkey = ":".join((str(rekey), str(key))) if rekey else key
            if value_type == dict:
                config[key] = self._set_arguments(value, args, setkey)
            if setkey in args:
                if args[setkey] is not None:
                    config[key] = args[setkey]
                del args[setkey]
            if value_type == str and value.lower() == "none":
                config[key] = None
        config.update(args)
        return config

    def _add_arguments(self, config, rekey=None):
        # add keys from config file to parser arguments
        for key, value in config.items():
            value_type = type(value)
            setkey = ":".join((str(rekey), str(key))) if rekey else key
            # list type
            if value_type == list:
                item_type = type(config[key][0]) if len(config[key]) else int
                self.add_argument(
                    "--" + str(setkey), required=False, nargs="+", type=item_type
                )
            # bool type
            elif value_type == bool:
                self.add_argument("--" + str(setkey), required=False, type=_str2value)
            # recursive add dict keys
            elif value_type == dict:
                self._add_arguments(config[key], rekey=key)
            elif value_type == str and value == "None":
                self.add_argument("--" + setkey, required=False, type=_str2value)
            # int, float, string type
            else:
                self.add_argument("--" + str(setkey), required=False, type=value_type)
