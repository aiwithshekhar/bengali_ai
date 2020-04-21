from dataset import BengaliDataset
from dataloader import Bengali_dataloader
from scores import *
import torch
import numpy as np
import random
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from tqdm import tqdm
from model_dispatcher import MODEL_DISPATCHER
from hyper import *
from collections import OrderedDict

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True

img_height = 137
img_width = 236
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


class Trainer:
    def __init__(self, model):

        self.num_epochs = 10
        self.phase = ["train", "val"]
        self.best_loss = float("inf")

        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self.device = torch.device("cuda:0")
        self.arc = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.loss = {name: [] for name in self.phase}

    def forward(self, image, targets):
        image = image.to(self.device)
        grapheme_root, vowel_diacritic, consonant_diacritic = targets
        t1 = grapheme_root.to(self.device)
        t2 = vowel_diacritic.to(self.device)
        t3 = consonant_diacritic.to(self.device)

        outputs = self.arc(image)

        o1, o2, o3 = outputs

        l1 = self.criterion(o1, t1)
        l2 = self.criterion(o2, t2)
        l3 = self.criterion(o3, t3)

        loss = (l1 + l2 + l3) / 3
        return loss, outputs

    def iterate(self, runner, run_args, epoch, phase):
        scr = Scores()
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase:{phase}| {start}")

        self.arc.train(phase == "train")

        d_loader = self.loader[phase]
        total_batches = len(d_loader)
        running_loss = 0.0
        print(run_args.batch_size, len(d_loader))
        # self.optimizer.zero_grad()
        runner.begin_epoch(phase)
        for itr, batch in tqdm(
            enumerate(d_loader), total=int(total_batches), position=0, leave=True
        ):
            image = batch["image"]
            grapheme_root = batch["grapheme_root"]
            vowel_diacritic = batch["vowel_diacritic"]
            consonant_diacritic = batch["consonant_diacritic"]

            targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
            loss, pred_targets = self.forward(image, targets)

            if phase == "train":
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            running_loss += (
                loss.item() * run_args.batch_size
            )  # running loss we are calculating for every batch and accumulating it
            scr.update(targets, pred_targets)
        epoch_loss = (
            running_loss / total_batches
        )  # so at the end we need to divide by the number of batches got get average loss per batch.
        epoch_sc = epoch_score(epoch_loss, scr)
        runner.end_epoch(epoch_loss, epoch_sc, phase)
        self.loss[phase].append(epoch_loss)

        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        params = OrderedDict(
            lr=[3e-2, 3e-4], batch_size=[88]
        )  # lr=[3e-2, 3e-4, 3e-6], batch_size=[64, 88, 128]
        m = RunManager()
        for run in RunBuilder.get_runs(params):
            self.optimizer = optim.Adam(self.arc.parameters(), lr=run.lr)
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="max", patience=5, factor=0.3, verbose=True
            )
            self.loader = {
                name: Bengali_dataloader(
                    name, img_height, img_width, mean, std, run.batch_size
                )
                for name in self.phase
            }

            m.begin_run(run, self.arc)

            for epoch in range(self.num_epochs):
                self.iterate(m, run, epoch, "train")
                state = {
                    "epoch": epoch,
                    "best_loss": self.best_loss,
                    "arc_state_dict": self.arc.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }
                with torch.no_grad():
                    val_loss = self.iterate(m, run, epoch, "val")
                    self.scheduler.step(val_loss)
                if val_loss < self.best_loss:
                    print(
                        "************New Region Found, Saving params****************************"
                    )
                    state["best_loss"] = self.best_loss = val_loss
                    torch.save(state, "checkpoints/weights_temp.pth")
            m.end_run()
        m.save("tune")


model_pass = MODEL_DISPATCHER["resnet34"](pretrained="imagenet")
run1 = Trainer(model_pass)
run1.start()
