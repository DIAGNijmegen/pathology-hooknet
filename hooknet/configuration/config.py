from argconfigparser.configuration.config import ApplicationConfiguration
from argconfigparser.configuration import Configurations
import pathlib
import os

class HookNetConfiguration(ApplicationConfiguration):

    NAME = "hooknet"

    CONFIG_PATH = pathlib.Path(__file__).absolute().parent / os.path.join(
        "config", "config.yml"
    )

    def __init__(
        self,
        default_key="default",
        resolve_keys=('model',),
        user_config=None,
        ext_settings=None,
        build_keys=("model",),
    ):
        super().__init__(
            pathlib.Path(self.__class__.CONFIG_PATH),
            default_key=default_key,
            resolve_keys=resolve_keys,
            user_config=user_config,
            ext_settings=ext_settings,
            build_keys=build_keys,
        )

def create_hooknet(user_config):

    application_configuration = ApplicationConfiguration(
        user_config=user_config, default_key='default', resolve_keys=('model',), build_keys=('model',)
    )
    print()
    hooknet_configuration = HookNetConfiguration(
        user_config=user_config, ext_settings=[application_configuration], build_keys=('model',)
    )
    return hooknet_configuration.get(('model'))