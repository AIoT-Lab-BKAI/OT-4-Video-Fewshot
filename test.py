import deepspeed
import torch
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset
from deepspeed.ops.adam import DeepSpeedCPUAdam
from pytorch_lightning.strategies import DeepSpeedStrategy
import torch.nn as nn
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_h = torch.nn.Linear(32, 32)
        self.layer = torch.nn.Linear(32, 2)
    
    def forward(self, x):
        x = deepspeed.checkpointing.checkpoint(self.layer_h, x)
        return self.layer(x)

class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = TestModel()


    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return DeepSpeedCPUAdam(self.model.parameters(), lr=0.001)


def run():
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    model = BoringModel()
    trainer = Trainer(
        devices = 1,
        accelerator='gpu',
        strategy=DeepSpeedStrategy(stage=3, offload_optimizer=True,offload_parameters=True, partition_activations=True, cpu_checkpointing=True),
    )
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)


if __name__ == "__main__":
    run()