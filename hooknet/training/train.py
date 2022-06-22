from tqdm import tqdm


MODES = ["training", "validation"]


class KerasUpdateLearningRate:
    def __init__(self, model, decay_rate=0.5, decay_steps=[2, 4, 1000, 5000, 10_000, 50_000, 100_000]):
        self._model = model
        self._lr = self._model.optimizer.lr.numpy()
        self._decay_rate = decay_rate
        self._decay_steps = decay_steps
        self._index = 0

    def __call__(self):
        if self._index in self._decay_steps:
            self._lr *= self._decay_rate
            self._model.optimizer.lr.assign(self._lr)
            print(f"updated learning rate={self._lr}")
        self._index += 1
        


def train(
    hooknet, epochs, steps, iterators, model_functions, metrics, tracker, weights_file
):

    best_metric = None
    update_learning_rate = KerasUpdateLearningRate(hooknet)
    for _ in tqdm(range(epochs)):
        for mode in MODES:
            for _ in range(steps):
                x_batch, y_batch, _ = next(iterators[mode])
                x_batch = list(x_batch.transpose(1, 0, 2, 3, 4))
                y_batch = y_batch.transpose(1, 0, 2, 3, 4)[0]
                
                if mode == 'training':
                    out = model_functions[mode](x=x_batch, y=y_batch, return_dict=True)
                else:
                    predictions = model_functions[mode](x=x_batch, argmax=False)
                    out = {'predictions': predictions, 'y': y_batch}
            
                for metric in metrics[mode]:
                    metric.update(**out)
                    
                update_learning_rate()

            metrics_data = {}
            for metric in metrics[mode]:
                metrics_data.update(metric())

            mode_metrics = {
                mode + "_" + name: value for name, value in metrics_data.items()
            }
            tracker.update(mode_metrics)

            if mode == "validation":
                # check if model improved
                if best_metric is None or metrics_data["F1_Macro"] > best_metric:
                    # update best metric
                    best_metric = metrics_data["F1_Macro"]
                    print("new best metric: ", best_metric)
                    # tracker.update_best(best_metric)
                    # save weights
                    print(f"Saving weights to: {weights_file}")
                    hooknet.save_weights(weights_file)


