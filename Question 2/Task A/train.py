import numpy as np
import torch
import matplotlib.pyplot as plt

def train(model, n_epochs, loss_fn, optim, train_dataloader, val_dataloader, device):

    training_losses = []
    training_accuracy = []
    val_losses = []
    val_accuracy = []
    
    for epoch in range(n_epochs):
        
        running_train_loss = 0.0
        correct_preds = 0
        total_preds = 0
    
        ''' Training '''
        
        for batch_id, (images, labels) in enumerate(train_dataloader):
    
            images = images.to(device) 
            labels = labels.to(device) 
            
            labels_one_hot = torch.zeros(labels.size(0), 10, device=device)
            labels_one_hot.scatter_(1, labels.view(-1, 1), 1) 
            
            pred = model(images)
            loss = loss_fn(pred, labels_one_hot)  
            running_train_loss += loss.item()
    
            _, predicted_classes = torch.max(pred, 1) 
            correct_preds += (predicted_classes == labels).sum().item()  
            total_preds += labels.size(0)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
    
        epoch_train_loss = running_train_loss/len(train_dataloader)
        epoch_train_acc = correct_preds/total_preds
    
        training_losses.append(epoch_train_loss)
        training_accuracy.append(epoch_train_acc)
    
        ''' Validation '''
    
        running_val_loss = 0.0
        correct_preds = 0
        total_preds = 0
    
        for batch_id, (images, labels) in enumerate(val_dataloader):
    
            images = images.to(device) 
            labels = labels.to(device) 
            
            labels_one_hot = torch.zeros(labels.size(0), 10, device=device)
            labels_one_hot.scatter_(1, labels.view(-1, 1), 1) 
            
            pred = model(images)
            loss = loss_fn(pred, labels_one_hot)  
            running_val_loss += loss.item()
    
            _, predicted_classes = torch.max(pred, 1) 
            correct_preds += (predicted_classes == labels).sum().item()  
            total_preds += labels.size(0)
    
        epoch_val_loss = running_val_loss/len(val_dataloader)
        epoch_val_acc = correct_preds/total_preds
    
        val_losses.append(epoch_val_loss)
        val_accuracy.append(epoch_val_acc)
        
        print(f'Epoch[{epoch}/{n_epochs}] - Train loss={epoch_train_loss}  Train acc={epoch_train_acc}  Val loss={epoch_val_loss}  Val acc={epoch_val_acc}')

    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].plot([i for i in range(n_epochs)],training_accuracy, label='Train')
    axs[0].plot([i for i in range(n_epochs)],val_accuracy, label='Valid')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('Accuracy v/s Epochs Curve')
    axs[0].legend()

    axs[1].plot([i for i in range(n_epochs)],training_losses, label='Training loss')
    axs[1].plot([i for i in range(n_epochs)],val_losses, label='Val loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Loss v/s Epochs Curve')
    axs[1].legend()
    plt.show()

    return max(training_accuracy), max(val_accuracy)
