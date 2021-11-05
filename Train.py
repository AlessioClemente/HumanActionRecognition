
import torch

from scipy import ndimage as ndimage            
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def batch(tensor, batch_size=64):
    tensor_list = []
    length = tensor.shape[0]
    i = 0
    while True:
        if (i + 1) * batch_size >= length:
            tensor_list.append(tensor[i * batch_size: length])
            return tensor_list
        tensor_list.append(tensor[i * batch_size: (i + 1) * batch_size])
        i += 1


def get_accuracy(model, x, y_ref):
    accuracy = 0.
    model.eval()
    with torch.no_grad():
        predicted = model(x)
        _, predicted = predicted.max(dim=1)
        accuracy = 1.0 * (predicted == y_ref).sum().item() / y_ref.shape[0]
    return accuracy, precision_score(predicted,y_ref,average='macro'), recall_score(predicted,y_ref,average='macro'),f1_score(predicted,y_ref,average='macro')




def Train(model, criterion, optimizer,
          x_train, y_train, x_test, y_test,
          force_cpu=False, num_epochs=200):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    x_train, y_train, x_test, y_test = x_train.to(device), y_train.to(device), x_test.to(device), y_test.to(device)
    

    # Prepare all mini-batches
    x_train_batches = batch(x_train)
    y_train_batches = batch(y_train)
    

    for epoch in range(num_epochs):

        model.train()

        current_loss = 0.0

        for  train_batches in zip(x_train_batches, y_train_batches):
            # get a mini-batch of sequences
            x_train_batch, y_train_batch = train_batches
            # zero the gradient parameters
            optimizer.zero_grad()

            # forward
            outputs = model(x_train_batch)


            loss = criterion(outputs, y_train_batch)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
        
        train_accuracy,train_precision,train_recall,train_f1 = get_accuracy(model, x_train, y_train)
        test_accuracy,test_precision,test_recall,test_f1 = get_accuracy(model, x_test, y_test)

        print('Epoch #{:03d} | Loss : {:.4e} | Accuracy_train : {:.4e} | Precision_train : {:.4e} | Recall_train : {:.4e} | F1Macro_train : {:.4e}| Accuracy_test : {:.4e} | Precision_test : {:.4e} | Recall_test : {:.4e} | F1Macro_test : {:.4e} '.format(
                epoch + 1, current_loss, train_accuracy,train_precision,train_recall,train_f1 ,test_accuracy,test_precision,test_recall,test_f1)) 

