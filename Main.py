import torch
from scipy import ndimage as ndimage            
from sklearn.utils import shuffle
import Network as n
import UTKinect as dataset
import Train as tr

n_chann = 216#186
train_set, test_set, train_labels, test_labels = dataset.to_Train(n_chann)
# Network instantiation
model = n.Network(n_channels=n_chann, n_classes=49, dropout_probability=0.2)

# Loss function & Optimizer
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.002,weight_decay=1e-5) 




num_epochs = 200

tr.Train(model=model, criterion=criterion, optimizer=optimizer,
      x_train=train_set, y_train=train_labels, x_test=test_set, y_test=test_labels,
      num_epochs=num_epochs)

with torch.no_grad():
    c = 0
    i = 0
    for i in range(0, len(test_set)-2, 3):
        batch = test_set[i:i+3]
        predictions = model(batch)
        _, predictions = predictions.max(dim=1)
        if(predictions.tolist()[0]==test_labels[i]):
            print(test_labels[i])
            c+=1
        if(predictions.tolist()[1]==test_labels[i+1]):
            print(test_labels[i+1])
            c+=1
        if(predictions.tolist()[2]==test_labels[i+2]):
            c+=1
            print(test_labels[i+2])

        
    print("Giusti {} su {}".format(c, len(test_labels)))
    print(str(int(100*c/len(test_labels))) + "%")
