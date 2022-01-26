from creationism.configuration.config import Configuration
from creationism.configuration.extensions import SEARCH_PATHS, open_config
import pathlib
import os


_DEFAULT_MODES = ("training", 'validation', 'testing')


class HookNetConfiguration(Configuration):

    NAME = "hooknet"

    CONFIG_PATH = pathlib.Path(__file__).absolute().parent / os.path.join(
        "config_files", "config.yml"
    )
    PRESETS_FOLDER = pathlib.Path(__file__).absolute().parent / "presets"
    SEARCH_PATHS = ("", pathlib.Path(__file__).absolute().parent / "config_files")

    def __init__(self, modes=_DEFAULT_MODES, search_paths=()):

        search_paths_ = self.__class__.SEARCH_PATHS + search_paths
        config_value = open_config(
            HookNetConfiguration.CONFIG_PATH, search_paths=search_paths_
        )

        super().__init__(
            name=HookNetConfiguration.NAME,
            modes=modes,
            config_value=config_value,
            search_paths=search_paths,
        )


def create_hooknet(user_config, mode):
    return HookNetConfiguration.build(user_config=user_config, modes=(mode,))[
        HookNetConfiguration.NAME
    ][mode]["model"]
