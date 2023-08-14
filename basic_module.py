import pytorch_lightning as pl
from deepspeed.ops.adam import DeepSpeedCPUAdam

from utils.utils import accuracy, criterion, mean_confidence_interval
def preprocess(batch):
    batch = list(batch)
    for i in range(len(batch)):
        batch[i] = batch[i].squeeze(0)
    return batch

class BasicModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sp_set, sp_labels, q_set):
        return self.model(sp_set, sp_labels, q_set)
        
    def training_step(self, batch, batch_idx):
        batch = preprocess(batch)
        sp_set, sp_labels, q_set, q_labels = batch
        
        pred_logits = self(sp_set, sp_labels, q_set)
        
        q_labels = q_labels.to(pred_logits.device)
        res = criterion(pred_logits, q_labels)

        self.log('train_loss', res['loss'], on_step=True, prog_bar=True)
        self.log('train_acc', res['acc'], on_step=True, prog_bar=True)

        return res['loss']
    
    def validation_step(self, batch, batch_idx):
        batch = preprocess(batch)
        sp_set, sp_labels, q_set, q_labels = batch

        pred_logits = self(sp_set, sp_labels, q_set)
        
        q_labels = q_labels.to(pred_logits.device)
        res = criterion(pred_logits, q_labels)
        
        self.log('val_loss', res['loss'], on_epoch=True, batch_size=1, prog_bar=True)
        self.log('val_acc', res['acc'], on_epoch=True, batch_size=1, prog_bar=True)
    
    def on_test_start(self):
        self.accuracies = []

    def on_train_start(self):
        # torch.cuda.empty_cache()
        pass
    
    def on_test_end(self):
        avg_acc, confidence_interval = mean_confidence_interval(self.accuracies)
        print(f'Accuracy: {avg_acc} +- {confidence_interval}')

    def test_step(self, batch, batch_idx):
        batch = preprocess(batch)
        sp_set, sp_labels, q_set, q_labels = batch        
        
        pred_logits = self(sp_set, sp_labels, q_set)
        q_labels = q_labels.to(pred_logits.device)

        acc = accuracy(pred_logits, q_labels, calc_mean=False)
        self.accuracies += acc.tolist()
    
    def configure_optimizers(self):
        optimizer = DeepSpeedCPUAdam(self.model.parameters())

        return optimizer