# User Defined Functions -- Utilities for Modeling




# Imports for Transfer Learning
import os
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Training Function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    
    # Training Vars
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training_start_time = time.time()
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        # Print out Epoch Info
        epoch_string = 'Epoch {} of {}'.format(epoch + 1, num_epochs)
        print(epoch_string) #add 1 since py starts @ 0
        print('-' * len(epoch_string))

        # Each epoch has a training and validation phase -- need rewrite using cifar-10 data dir
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data -- Needs rewrite for dict style
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs) #do not care if is inception model or not (resnet18) -- deleted a portion of inception specific code
                    loss = criterion(outputs, labels)

                    pred_likelihood, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        #loss = Variable(loss, requires_grad=True, dtype=torch.float) #tried fixing issue with loss not being used for backprop
                        loss.backward()
                        optimizer.step()


                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} | Accuracy: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Deep Copy of best Model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - training_start_time
    print('Total Training Time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))
    print(f'Best Test Acc: {100*best_acc : .2f}%')

    # load best model weights
    #return the most confident preds for correct & incorrect! 
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# Data Visualization Function (top 5 preds per class -- correct/incorrect)
def viz_preds_by_confidence_for_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    
    # Training Vars
    training_start_time = time.time()
    val_acc_history = []
    
    #These preds are per class, so 50 & 50 for each!
    t5_correct_preds = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    t5_wrong_preds = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    '''
    # One Dict for both Correct & Wrong! 
    {
        0:[tensor, tensor, tensor, tensor, tensor], 
        1:[tensor, tensor, tensor, tensor, tensor], 
        2:[tensor, tensor, tensor, tensor, tensor], 
        3:[tensor, tensor, tensor, tensor, tensor], 
        4:[tensor, tensor, tensor, tensor, tensor], 
        5:[tensor, tensor, tensor, tensor, tensor], 
        6:[tensor, tensor, tensor, tensor, tensor], 
        7:[tensor, tensor, tensor, tensor, tensor], 
        8:[tensor, tensor, tensor, tensor, tensor], 
        9:[tensor, tensor, tensor, tensor, tensor]
    }

    1 tensor in dict! ([input_data], [true_label, is_correct_pred (T/F), predicted_class, confidence_in_prediction])
    '''


    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        # Print out Epoch Info
        epoch_string = 'Epoch {} of {}'.format(epoch + 1, num_epochs)
        print(epoch_string) #add 1 since py starts @ 0
        print('-' * len(epoch_string))

        # Each epoch has a training and validation phase -- need rewrite using cifar-10 data dir
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data -- Needs rewrite for dict style
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs) #do not care if is inception model or not (resnet18) -- deleted a portion of inception specific code
                    loss = criterion(outputs, labels)

                    pred_likelihood, preds = torch.max(outputs, 1)
                    #returns the index of the max value in each output dim (of 10 class output, returns the index of most likely output value!)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Get Top 5 Correct & Incorrect Preds
                correct_predictions = preds.eq(labels.data) #corrects are True or 1 (try to make last element of obj to be added to dicts!)
                #pred_likelihood, pred_indicies = torch.max(outputs) #likelihood of the prediction class (confidence)
                
                t5_meta = torch.stack((labels, correct_predictions, preds, pred_likelihood), dim=-1) #combines the 4 metadata values
                input_data = inputs

                # Get Data into List of Lists, will check confidence in next block
                t5_tensors = []
                #print("num labels: ", len(labels)) #del later -- just want to visualize
                for i in range(0, len(labels)):
                    input_data_instance = input_data[i, :, :, :]
                    
                    t5_tensors.append([input_data_instance, t5_meta[i]])
                    #print("len of t5 tensors- ", len(t5_tensors))


                # Iterate through concated tensor and find top 5 correct/incorrects (update/append if possible)
                for instance in t5_tensors: 
                    prediction_is_correct = instance[1][1]
                    class_of_pred = which_class(instance) #func below, needed ints for the dict keys
                    confidence_in_prediction = instance[1][3]
                    
                    if prediction_is_correct == 1.0:
                        #print("pred_class:", class_of_pred)
                        if len(t5_correct_preds[class_of_pred]) < 5:
                            t5_correct_preds[class_of_pred].append(instance)
                        else:
                            for i in t5_correct_preds[class_of_pred]:
                                if torch.equal(instance[0], i[0]) != True:
                                    if i[1][3] < instance[1][3]:
                                      t5_correct_preds[class_of_pred].remove(i)
                                      t5_correct_preds[class_of_pred].append(instance)
                                      break
                                break

                                    #t5_correct_preds[class_of_pred] = [instance if i[1][3]<instance[1][3] else i for i in t5_correct_preds[class_of_pred]]

                    elif prediction_is_correct == 0.0: #incorrect pred
                        if len(t5_wrong_preds[class_of_pred]) < 5:
                            t5_wrong_preds[class_of_pred].append(instance)
                        else:
                            for i in t5_wrong_preds[class_of_pred]:
                                if torch.equal(instance[0], i[0]) != True:
                                    if i[1][3] < instance[1][3]:
                                      t5_wrong_preds[class_of_pred].remove(i)
                                      t5_wrong_preds[class_of_pred].append(instance)
                                      break
                                break
                # Check num of values per key after iters -- works!
                #for key in t5_correct_preds:
                #    print(f"for key{key} - len of vals={len(t5_correct_preds[key])}")


                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} | Accuracy: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Deep Copy of best Model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - training_start_time
    print('Total Training Time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))
    print(f'Best Test Acc: {100*best_acc : .2f}%')

    # load best model weights
    #return the most confident preds for correct & incorrect! 
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, t5_correct_preds, t5_wrong_preds


# Function that changes `.requires_grad` Attribute to false (if Feature Extracting!)
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# Function that identifies params that need to be updated (Fine Tune Vs. Feature Extraction)
def params_to_learn(feature_extract, model_ft, params_to_update):
    print("Params to learn:") #verified that only our output layer params are to be updated!
    if feature_extract: #is True
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)


# Function to determine which class the instance belongs to (used in visualization func, need int dtype for key)
def which_class(instance):
    inst_class = instance[1][0]
    inst_class = int(inst_class)
    return inst_class


# Show a Single Img
def show_img(inp, img_label=None):
    """Imshow for Tensor."""
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    
    plt.xticks([])
    plt.yticks([])
    plt.imshow(inp)
    if img_label is not None:
        #plt.title(img_label) #switched to the xlabels, not title
        plt.xlabel(img_label)
    plt.pause(0.001)  # pause a bit so that plots are updated


