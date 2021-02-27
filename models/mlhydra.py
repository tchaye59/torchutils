from models.base import *
from torch import nn


class MLHydra(nn.Module):
    """ """

    def __init__(self, backbone: BaseModel, heads: List[BaseModel], verbosity=10):
        super().__init__()
        assert len(heads), f'Empty heads'
        self.backbone = backbone
        self.heads = heads
        self.verbosity = verbosity

    def forward(self, X):
        _, X = self.backbone(X)
        pred = [head(X) for head in self.heads]
        return torch.stack(pred, 1).squeeze(-1)

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self(X)

    def fit_backbone(self, train_loader, epochs=1, val_loader=None):
        # set trainable
        print('Training backbone:')
        return self.backbone.fit(train_loader, epochs, val_loader)

    def fit_heads(self, train_loader, epochs=1, val_loader=None, callbacks: List[Callback] = []):

        self.callbacks = callbacks
        [callback.set_model(self) for callback in callbacks]

        # disable gradient on backbone
        print('Training heads:')

        history = {}

        train_steps = len(train_loader)
        val_steps = len(val_loader) if val_loader is not None else 0

        [[callback.on_train_begin() for callback in head.callbacks] for head in self.heads]
        for epoch in range(epochs):
            epoch_info_sum = {}

            print(f'Epoch: {epoch + 1}/{epochs}:')

            # Training
            [[callback.on_epoch_begin(epoch) for callback in head.callbacks] for head in self.heads]
            for batch_idx, batch in enumerate(train_loader):
                X, y = to_device(batch)
                with torch.no_grad():
                    _, X = self.backbone(X)
                X = X.detach()
                for head_idx, head_model in enumerate(self.heads):
                    label = (y == head_idx).unsqueeze(1).float()
                    batch = X, label
                    info = head_model.training_step(batch)
                    self.update_history(history, epoch_info_sum, info, batch_idx, train_steps, head_idx)
                    if batch_idx % self.verbosity == 0 or batch_idx == train_steps - 1:
                        print(
                            f'Training: {batch_idx + 1}/{train_steps}  {epoch_info_to_string(epoch_info_sum, batch_idx)}',
                            end='\r',
                            file=sys.stdout,
                            flush=True)

            [[callback.on_train_end(history) for callback in head.callbacks] for head in self.heads]

            # Validation
            if val_loader is not None:
                epoch_info_sum = {}
                print()
                [[callback.on_test_begin(epoch, ) for callback in head.callbacks] for head in self.heads]
                for batch_idx, batch in enumerate(val_loader):
                    X, y = to_device(batch)
                    _, X = self.backbone(X)
                    X = X.detach()
                    for head_idx, head_model in enumerate(self.heads):
                        head_model = self.heads[head_idx]
                        label = (y == head_idx).unsqueeze(1).float()
                        batch = X, label
                        info = head_model.validation_step(batch)
                        self.update_history(history, epoch_info_sum, info, batch_idx, val_steps, head_idx)

                    if batch_idx % self.verbosity == 0 or batch_idx == val_steps - 1:
                        print(
                            f'Validation: {batch_idx + 1}/{val_steps}  {epoch_info_to_string(epoch_info_sum, batch_idx)}',
                            end='\r',
                            file=sys.stdout, flush=True)

                [[callback.on_test_end(history) for callback in head.callbacks] for head in self.heads]

            [[callback.on_epoch_end(epoch, history) for callback in head.callbacks] for head in self.heads]
            print()

        return history

    def update_history(self, history, epoch_info_sum, info, batch_idx, train_steps, head_idx):
        for key in info:
            tmp_key = f'{key}{head_idx + 1}'
            val = info[key].detach()
            ss = epoch_info_sum.get(tmp_key, 0) + val
            epoch_info_sum[tmp_key] = ss
            if history is not None and batch_idx + 1 == train_steps:
                data = history.get(tmp_key, [])
                ss = float((ss / (batch_idx + 1)).cpu().numpy())
                data.append(ss)
                history[tmp_key] = data

    def evaluate(self, data_loader, ):
        history = {}
        history_sum = {}
        steps = len(data_loader)
        for batch_idx, batch in enumerate(data_loader):
            batch = to_device(batch)
            info = self.validation_step(batch)
            self.update_history(history, history_sum, info, batch_idx, steps)
            if batch_idx % self.verbosity == 0:
                print(f'Evaluate: {batch_idx + 1}/{steps}  {epoch_info_to_string(history_sum, batch_idx)}',
                      end='\r',
                      file=sys.stdout,
                      flush=True)
        print()
        return history
