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
        self.lr = 1e-4
        self.batch_size = {"train": 128, "val": 64}
        # self.num_workers = 6
        self.num_epochs = 20
        self.phase = ["train", "val"]
        self.best_loss = float("inf")

        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        # cudnn.benchmark = True
        self.device = torch.device("cuda:0")

        self.arc = model.to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.arc.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", patience=5, factor=0.3, verbose=True
        )

        self.loader = {
            name: Bengali_dataloader(
                name, img_height, img_width, mean, std, self.batch_size[name],
            )
            for name in self.phase
        }
        self.loss = {name: [] for name in self.phase}

    def forward(self, image, targets):
        image = image.to(self.device)
        grapheme_root, vowel_diacritic, consonant_diacritic = targets
        t1 = grapheme_root.to(self.device)
        t2 = vowel_diacritic.to(self.device)
        t3 = consonant_diacritic.to(self.device)

        outputs = self.arc(image)

        o1, o2, o3 = outputs

        # print(f"outputs {o1.shape} {o2.shape} {o3.shape}")
        # print(f"targets {t1.shape} {t2.shape} {t3.shape}")

        l1 = self.criterion(o1, t1)
        l2 = self.criterion(o2, t2)
        l3 = self.criterion(o3, t3)

        loss = (l1 + l2 + l3) / 3
        return loss, outputs

    def iterate(self, epoch, phase):
        scr = Scores()
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase:{phase}| {start}")

        self.arc.train(phase == "train")
        batch_size = self.batch_size[phase]
        d_loader = self.loader[phase]
        total_batches = len(d_loader)
        running_loss = 0.0
        # self.optimizer.zero_grad()

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
                loss.item()
            )  # running loss we are calculating for every batch and accumulating it
            scr.update(targets, pred_targets)
        epoch_loss = (
            running_loss / total_batches
        )  # so at the end we need to divide by the number of batches got get average loss per batch.
        self.loss[phase].append(epoch_loss)
        epoch_score(epoch_loss, scr)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "arc_state_dict": self.arc.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print(
                    "************New Region Found, Saving params****************************"
                )
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "checkpoints/weights.pth")


model_pass = MODEL_DISPATCHER["resnet34"](pretrained="imagenet")
run1 = Trainer(model_pass)
run1.start()
