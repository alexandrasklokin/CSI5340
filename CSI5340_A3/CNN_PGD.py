#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tensorflow as tf
import torchvision
import torch

from sklearn.metrics import accuracy_score

from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

LOSS_FUNCTION = nn.CrossEntropyLoss()

# ___________________________________________________________________________________________________________________ HELPER FUNCTIONS

def displayMNIST(data, index, title="Title", save=False):
    
    plt.imshow(data.data[index], cmap='gray')
    
    if save: plt.savefig(title+".png")
    
    plt.show()

def displayImage(data, index, title="Title", save=False):
    
    data = [item.detach().numpy() for item in data]
    
    plt.imshow(data[index].reshape(28,28), cmap='gray')
    
    if save: plt.savefig(title+".png")
    
    plt.show()


def getLabel(data, index):
    
    return data.targets[index].numpy().item()


def plotAccuracy(accuracy, title="Title", save=False):
    
    plt.plot(range(len(accuracy)), accuracy, '-g', label='Accuracy')
    
    plt.legend(loc='upper left')
    
    if save: plt.savefig(title+"_Accuracy.png")

    plt.show()


def plotLoss(loss, title="Title", save=False):
    
    plt.plot(range(len(loss)), loss, '-r', label='Loss')
    
    plt.legend(loc='upper left')
    
    if save: plt.savefig(title+"_Loss.png")

    plt.show()


def removeTrueLabelFromTensor(T, i):
    T[i] = (-1000000000.0)
    return T


# ___________________________________________________________________________________________________________________ CNN IMPLEMENTATION


class CNN (nn.Module) :
    
    def __init__(self):
        
        super(CNN, self).__init__()
        
        self.layer1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=8,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.out = nn.Linear(2304, 10)
    
    def forward(self, x):
        
        x = self.layer1(x)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization


def getLossAccuracy(cnn_model, data_load, dataset, details=False):

    cnn_model.eval()
    
    with torch.no_grad():
        
        correct = 0
        total = 0
        
        for data, label in data_load[dataset]:
            
            output_label, last_layer = cnn_model(data)
            pred_y = torch.max(output_label, 1)[1].data.squeeze()
            
            accuracy = (pred_y == label).sum().item() / float(label.size(0))
            loss = LOSS_FUNCTION(output_label, label).item()
    
    if details: print(dataset+' loss: %.5f' % loss)
    if details: print(dataset+' accuracy: %.5f' % accuracy)
        
    return (loss, accuracy)

def train(cnn_model, data_load, optim, EPOCHS):
    
    training_accuracy = []
    training_loss = []
    
    cnn_model.train()
    
    # Initial Training Loss and Accuracy
    (tr_loss, tr_accuracy) = getLossAccuracy(cnn_model, data_load, 'train')            
    training_loss.append(tr_loss)
    training_accuracy.append(tr_accuracy)

    # Train the model
    total_step = len(data_load['train'])

    for epoch in range(EPOCHS):
        for i, (training_data, training_label) in enumerate(data_load['train']):
            
            batch_data = Variable(training_data)
            batch_label = Variable(training_label) 
            output_label, _ = cnn_model(batch_data)               
            
            loss = LOSS_FUNCTION(output_label, batch_label)
  
            optim.zero_grad()           
            loss.backward()             
            optim.step()                

            if (i+1) % 200 == 0:
                print ('Epoch [{}/{}], Progress [{}/{}], Loss: {:.4f}' .format(epoch + 1, EPOCHS, i + 1, total_step, loss.item()))
            
        
        # GET LOSS AND ACCURACY AT EVERY EPOCH
        (tr_loss, tr_accuracy) = getLossAccuracy(cnn_model, data_load, 'train')            
        training_loss.append(tr_loss)
        training_accuracy.append(tr_accuracy)
        #print ('      Training Loss: {:.4f}, Training Accuracy: {:.4f}' .format(tr_loss, tr_accuracy))
            
    return (training_loss, training_accuracy)


# ___________________________________________________________________________________________________________________ UNTARGETED PGD


def getLossAccuracyAdversarial(cnn_model, adversarial_imgs, true_labels, details=False):
    
    true_labels = true_labels.type(torch.int64)
    
    cnn_model.eval()
    
    with torch.no_grad():
        
        correct = 0
        total = 0
        
        output_label, last_layer = cnn_model(adversarial_imgs)
        pred_y = torch.max(output_label, 1)[1].data.squeeze()
        
        #print(pred_y[0:20])
        #print(true_labels[0:20])
        
        accuracy = (pred_y == true_labels).sum().item() / float(true_labels.size(0))
        loss = LOSS_FUNCTION(output_label, true_labels).item()
    
    
    if details: print(dataset+' loss: %.5f' % loss)
    if details: print(dataset+' accuracy: %.5f' % accuracy)
        
    return (loss, accuracy)


def untargeted_pgd_attack(cnn_model, images, true_labels, eps, alpha, iterations):
    
    original_images = images.data
    
    for j in range(iterations):

        images.requires_grad = True
        (pred_label, _) = cnn_model(images)
        
        loss = LOSS_FUNCTION(pred_label, true_labels)
        loss.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-eps, max=eps)
        images = torch.clamp(original_images + eta, min=0, max=1).detach_()
            
    return (loss, images, true_labels)


def pgd_train(cnn_model, data_load, optim, pgd_function, EPOCHS, eps, alpha, iterations):
    
    training_accuracy = []
    training_loss = []
    adv_training_loss = []
    adv_training_accuracy = []
    
    cnn_model.train()
    
    # Initial Training Loss and Accuracy
    (tr_loss, tr_accuracy) = getLossAccuracy(cnn_model, data_load, 'train')
    training_loss.append(tr_loss)
    adv_training_loss.append(tr_loss)
    training_accuracy.append(tr_accuracy)
    adv_training_accuracy.append(tr_accuracy)

    # Train the model
    total_step = len(data_load['train'])

    for epoch in range(EPOCHS):
        
        original_images = torch.empty(0,)
        adversarial_images = torch.empty(0,)
        adversarial_labels = torch.empty(0,)
        
        for i, (training_data, training_label) in enumerate(data_load['train']):            
            
            # PGD ATTACK TO PRODUCE ADVERSARIAL EXAMPLES ON THIS BATCH
            (pgd_loss, perturbed_imgs, pgd_label) = pgd_function(cnn_model, training_data, training_label, eps, alpha, iterations)
            
            # COLLECT ALL ADVERSARIAL EXAMPLES
            original_images = torch.cat((original_images, training_data),0)
            adversarial_images = torch.cat((adversarial_images, perturbed_imgs),0)
            adversarial_labels = torch.cat((adversarial_labels, pgd_label),0)
            
            # TRAIN MODEL BASED ON PERTURBED DATA AND UNTARGETTED LABEL
            batch_data = Variable(perturbed_imgs)
            batch_label = Variable(pgd_label)
            
            # PERFORM THE PREDICTION ON CURRENT DATA
            output_label, _ = cnn_model(batch_data)             
            
            # LOSS FUNCTION
            batch_training_loss = LOSS_FUNCTION(output_label, batch_label)
  
            # CLEAR GRADIENTS
            optim.zero_grad() 
        
            # CALCULATE GRADIENTS
            batch_training_loss.backward()   
            
            # UPDATE MODEL PARAMS
            optim.step()                

            if (i+1) % 200 == 0:
                print ('Epoch [{}/{}], Progress [{}/{}], Loss: {:.4f}' .format(epoch + 1, EPOCHS, i + 1, total_step, batch_training_loss))
        
        # GET LOSS AND ACCURACY AT EVERY EPOCH (ON PERTURBED DATA)            
        (tr_loss, tr_accuracy) = getLossAccuracyAdversarial(cnn_model, perturbed_imgs, pgd_label)
        adv_training_loss.append(tr_loss)
        adv_training_accuracy.append(tr_accuracy)
        #print ('      Adversarial Loss: {:.4f}, Adversarial Accuracy: {:.4f}' .format(tr_loss, tr_accuracy))
        
        # GET LOSS AND ACCURACY AT EVERY EPOCH (ON ORIGINAL TRAINING DATA)
        (tr_loss, tr_accuracy) = getLossAccuracy(cnn_model, data_load, 'train')            
        training_loss.append(tr_loss)
        training_accuracy.append(tr_accuracy)
        #print ('      Training Loss: {:.4f}, Training Accuracy: {:.4f}' .format(tr_loss, tr_accuracy))
            
    return (original_images, adversarial_images, adversarial_labels, 
            training_loss, training_accuracy, adv_training_loss, adv_training_accuracy)


# ___________________________________________________________________________________________________________________ TARGETED PGD


def targeted_pgd_attack(cnn_model, images, true_labels, eps, alpha, iterations, details=False):
    
    original_images = images.data
    
    for j in range(iterations):

        images.requires_grad = True
        (pred_label, _) = cnn_model(images)
        
        loss = LOSS_FUNCTION(pred_label, true_labels)
        loss.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-eps, max=eps)
        images = torch.clamp(original_images + eta, min=0, max=1).detach_()
        
    if details: print("____________________________________________ BEFORE PERTURBATION")
    if details: displayImage(original_images, 0)
    if details: print("TRUE LABEL:" + str(true_labels[0]))
        
    targeted_label, _ = cnn_model(images)
    
    if details: print("____________________________________________ AFTER PERTURBATION (TARGETTED)")
    if details: displayImage(images, 0)
    if details: print("CNN RESULT:" + str(targeted_label[0]))
    
    # DO NOT ALLOW TO CHOOSE TRUE LABEL
    targeted_label = [elem.detach().numpy() for elem in targeted_label]
    targeted_label = [removeTrueLabelFromTensor(targeted_label[i], true_labels[i]) for i in range(len(targeted_label))]
    targeted_label = torch.Tensor(targeted_label)
    targeted_label = torch.max(targeted_label, 1)[1].data.squeeze()
    
    if details: print("AKA: "+str(targeted_label[0]))
            
    return (loss, images, targeted_label)

# ___________________________________________________________________________________________________________________ TESTING PGD


def pgd_test(cnn_model, data_load, pgd_function, eps=0.3, alpha=0.01, iterations=1, details=False):
    
    (_, test_accuracy) = getLossAccuracy(cnn_model, data_load, 'test', details=False)
    
    if eps == 0:
        
        pgd_accuracy = test_accuracy
        
    else:        
        adversarial_images = torch.empty(0,)
        adversarial_labels = torch.empty(0,)
    
        total_step = len(data_load['test'])

        for i, (testing_images, true_value) in enumerate(data_load['test']): 
            
            (_, perturbed_imgs, pgd_label) = pgd_function(cnn_model, testing_images, true_value, eps, alpha, iterations)
            adversarial_images = torch.cat((adversarial_images, perturbed_imgs), 0)
            adversarial_labels = torch.cat((adversarial_labels, pgd_label), 0)
            
        (_, pgd_accuracy) = getLossAccuracyAdversarial(cnn_model, adversarial_images, adversarial_labels)
    
    if details: print('Accuracy on original Test Set: %.5f' % test_accuracy)
    if details: print('Accuracy of Perturbed Test Set: %.5f' % pgd_accuracy)
        
    return (test_accuracy, pgd_accuracy)
