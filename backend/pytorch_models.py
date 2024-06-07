import torch
from torch import nn
from torch.utils.data import DataLoader

class modelV0(nn.Module):
  def __init__(self, vocab_size, embed_dim):
    super().__init__()
    self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
    self.layer_1 = nn.Linear(in_features=128, out_features=100)
    self.layer_2 = nn.Linear(in_features=100, out_features=100)
    self.layer_3 = nn.Linear(in_features=100, out_features=6) # out = None, Stage 1, Stage 2, or Stage 3
    self.relu = nn.ReLU() # non-linear activation function
    self.init_weights()

  def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.layer_1.weight.data.uniform_(-initrange, initrange)
        self.layer_1.bias.data.zero_()
        self.layer_2.weight.data.uniform_(-initrange, initrange)
        self.layer_2.bias.data.zero_()
        self.layer_3.weight.data.uniform_(-initrange, initrange)
        self.layer_3.bias.data.zero_()

  def forward(self, x, offsets):
    return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(self.embedding(x, offsets))))))

class training():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  def train_step(self,
                model: torch.nn.Module,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                train_data,
                test_data,
                vocab,
                tokenizer):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    train_dataloader = DataLoader(train_data, batch_size=1,
                              shuffle=True, collate_fn=self.collate_batch)
    test_dataloader = DataLoader(test_data, batch_size=1,
                              shuffle=True, collate_fn=self.collate_batch)
    
    self.vocab = vocab
    self.tokenizer = tokenizer

    epochs = 10

    for epoch in range(epochs):
        for idx, (label, text, offsets) in enumerate(train_dataloader):
          y_train = torch.tensor([0, 0, 0, 0, 0, 0])
          y_train[label] = 1
          model.train()
          y_logits = model(text, offsets)
          y_preds = torch.round(torch.sigmoid(y_logits))
          loss = loss_fn(y_logits, label)
          acc = self.accuracy_fn(y_true = label, y_pred=y_preds)  
          optimizer.zero_grad()
          loss.backward()
          parameters = [p for p in model.parameters() if p.grad is not None]
          total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
          max_norm = 0.1
          if total_norm > max_norm:
              clip_coef = max_norm / (total_norm + 1e-6)
              for p in parameters:
                  p.grad.data.mul_(clip_coef)
          optimizer.step()

        model.eval()
        with torch.inference_mode():
            total_acc, total_count = 0, 0
            for idx, (label, text, offsets) in enumerate(test_dataloader):
              y_test = torch.tensor([0, 0, 0, 0, 0, 0])
              y_test[label] = 1
              test_logits = model(text, offsets)
              y_pred = torch.round(torch.sigmoid(test_logits))
              test_loss = loss_fn(test_logits, label)
              test_acc = self.accuracy_fn(y_pred=y_pred, y_true=label)
              total_acc += (test_logits.argmax(1) == label).sum().item()
              total_count += label.size(0)
              if idx % 500 == 0 and idx > 0:
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| accuracy {:8.3f}'.format(epoch, idx, len(test_dataloader),
                                                  total_acc/total_count))
                total_acc, total_count = 0, 0
        if(epoch % 1 == 0):
            print(f"Epoch: {epoch} | Loss: {loss:0.5f}, Acc: {acc:.2f}% | Test loss: {test_loss: .5f}, Test acc: {test_acc:.2f}%")
  def accuracy_fn(self, y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = correct/len(y_pred)
    return acc*100
  def collate_batch(self, batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(int(_label))
         processed_text = torch.tensor(self.vocab(self.tokenizer(_text)), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(self.device), text_list.to(self.device), offsets.to(self.device)