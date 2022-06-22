import wandb


class Tracker:
    def log_metric(self, metric, epoch, step, value):
        pass

    def update(self, updates):
        pass


class WandbTracker(Tracker):
    def __init__(self, project, log_path):
        wandb.init(project=project, dir=log_path)

    def update(self, updates):
        wandb.log(updates)

    def update_best(self, best_value):
        wandb.run.summary["best_metric"] = best_value

    def save(self, file):
        wandb.save(file)

    def update_config(self, config):
        wandb.config.update(config)