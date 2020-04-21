from collections import OrderedDict
from collections import namedtuple
from itertools import product
import time
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from IPython import display

# pd.set_option("display.max_columns", 8)


class RunBuilder:
    @staticmethod
    def get_runs(params):

        Run = namedtuple("Run", params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs


params = OrderedDict(lr=[0.01, 0.001], batch_size=[1000, 10000])


class RunManager:
    def __init__(self):

        self.epoch_count = 0
        self.epoch_loss = 0
        self.score = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.tb = None

    def begin_run(self, run, network):
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1

        self.network = network
        self.tb = SummaryWriter(comment=f"-{run}")

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self, phase):
        self.epoch_start_time = time.time()
        if phase == "train":
            self.epoch_count += 1

        self.epoch_loss = 0
        self.epoch_score = 0

    def end_epoch(self, epoch_loss, epoch_score, phase):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        if phase == "train":
            self.tb.add_scalar("Train_Loss", epoch_loss, self.epoch_count)
            self.tb.add_scalar("Train_Recall", epoch_score, self.epoch_count)
        else:
            self.tb.add_scalar("Val_Loss", epoch_loss, self.epoch_count)
            self.tb.add_scalar("Val_Recall", epoch_score, self.epoch_count)

        for name, param in self.network.named_parameters():
            p = name.split(".")
            if "last_linear" in p:
                pass
            else:
                self.tb.add_histogram(name, param, self.epoch_count)
                self.tb.add_histogram(f"{name}.grad", param.grad, self.epoch_count)

        results = OrderedDict()
        results["run"] = f"{phase}_{self.run_count}"
        results["epoch"] = self.epoch_count
        results["epoch_loss"] = epoch_loss
        results["epoch_score"] = epoch_score
        results["epoch_duration"] = epoch_duration
        results["run_duration"] = run_duration

        for k, v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient="columns")
        display.clear_output(wait=True)
        display.display(df)

    def save(self, fileName):
        pd.DataFrame.from_dict(self.run_data, orient="columns").to_csv(
            f"{fileName}.csv"
        )
