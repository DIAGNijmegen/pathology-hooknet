from pathlib import Path

import wandb
import yaml


class Tracker:
    def log_metric(self, metric, epoch, step, value):
        pass

    def update(self, updates):
        pass


class WandbTracker(Tracker):
    def __init__(self, project, log_path):
        self._log_path = Path(log_path)
        wandb.init(project=project, dir=log_path)

    def update(self, updates):
        wandb.log(updates)

    def update_best(self, best_value):
        wandb.run.summary["best_metric"] = best_value

    def save(self, data):
        if isinstance(data, dict):
            name = data.keys()[0] + '.yml'
            with open(self._log_path/name, 'w') as outfile:
                yaml.dump(data, outfile, default_flow_style=False)
            wandb.save(self._log_path/name)
        else:
            wandb.save(data)

    def update_config(self, config):
        wandb.config.update(config)