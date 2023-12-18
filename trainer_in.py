import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score
from utils import binary_cross_entropy


class Trainer_in(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader, args):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = args.epoch
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.best_model, self.best_epoch = None, None
        self.best_auroc = 0

        self.output_dir = args.res_dir




    def train(self):
        for i in range(self.epochs):
            self.current_epoch += 1
            
            train_loss = self.train_epoch()
            auroc, auprc, val_loss = self.test(dataloader="val")
            
            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch
                self.save_result()

            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), " AUROC "
                  + str(auroc) + " AUPRC " + str(auprc))
            
            

        auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision = self.test(dataloader="test")

        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " AUROC "
              + str(auroc) + " AUPRC " + str(auprc) + " Sensitivity " + str(sensitivity) + " Specificity " +
              str(specificity) + " Accuracy " + str(accuracy) + " Thred_optim " + str(thred_optim))

        
        return None

    def save_result(self):
        torch.save(self.best_model.state_dict(),
                    os.path.join(self.output_dir, f"best_model.pth"))

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for i, (v_d, v_p, labels) in enumerate(self.train_dataloader):

            v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device)
            
            self.optim.zero_grad()
            score = self.model(v_d, v_p)
            
            n, loss = binary_cross_entropy(score, labels)
            
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
            
            
            
        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch

    def test(self, dataloader="test"):
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            for i, (v_d, v_p, labels) in enumerate(data_loader):

                v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device)

                if dataloader == "val":
                    score = self.model(v_d, v_p)
                elif dataloader == "test":
                    score = self.best_model(v_d, v_p)
                n, loss = binary_cross_entropy(score, labels)
                
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()
                
                
                
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        test_loss = test_loss / num_batches

        if dataloader == "test":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            precision = tpr / (tpr + fpr)
            f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i >= thred_optim else 0 for i in y_pred]
            cm1 = confusion_matrix(y_label, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
            specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
            precision1 = precision_score(y_label, y_pred_s)
            return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, test_loss, thred_optim, precision1
        else:
            return auroc, auprc, test_loss
