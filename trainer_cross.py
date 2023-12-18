import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score
from utils import binary_cross_entropy, graph_collate_func

from torch.nn import functional as F


class Trainer_cross(object):
    def __init__(self, Models, Optims, device, multi_generator, val_dataloader, test_dataloader, args):
        self.optG, self.optC1, self.optC2 = Optims[0], Optims[1], Optims[2]
        self.G, self.C1, self.C2 = Models[0], Models[1], Models[2]
        self.device = device
        self.current_epoch = 0
        self.epochs = args.epoch
        self.train_dataloader = multi_generator
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        

        self.best_model_G, self.best_model_C1, self.best_model_C2, self.best_epoch = None, None, None, None
        self.best_auroc = 0

        self.output_dir = args.res_dir


    def train(self):
        for i in range(self.epochs):
            self.current_epoch += 1
            
            train_loss = self.train_epoch()
            auroc, auprc, val_loss = self.test(dataloader="val")
            
            if max(auroc) >= self.best_auroc:
                self.best_model_G = copy.deepcopy(self.G)
                self.best_model_C1 = copy.deepcopy(self.C1)
                self.best_model_C2 = copy.deepcopy(self.C2)
                self.best_auroc = max(auroc)
                self.best_epoch = self.current_epoch 
                self.save_result() 

            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), " AUROC "
                  + str(auroc) + " AUPRC " + str(auprc))
            
        return None


    def save_result(self):
        torch.save(self.best_model_G.state_dict(),
                    os.path.join(self.output_dir, f"best_model_G.pth"))
        torch.save(self.best_model_C1.state_dict(),
                    os.path.join(self.output_dir, f"best_model_C1.pth"))
        torch.save(self.best_model_C2.state_dict(),
                    os.path.join(self.output_dir, f"best_model_C2.pth"))

    def reset_grad(self):
        self.optG.zero_grad()
        self.optC1.zero_grad()
        self.optC2.zero_grad()

    def train_epoch(self):

        self.G.train()
        self.C1.train()
        self.C2.train()
        
        loss_epoch = [0, 0, 0]
        num_batches = len(self.train_dataloader)
        for i, (batch_s, batch_t) in enumerate(self.train_dataloader):
            
            v_d, v_p = batch_t[0].to(self.device), batch_t[1].to(self.device)
            sv_d, sv_p, slabels = batch_s[0].to(self.device), batch_s[1].to(self.device), batch_s[2].float().to(self.device)

            self.reset_grad()
            
            # step 1
            output_s1 = self.C1(self.G(sv_d, sv_p))
            output_s2 = self.C2(self.G(sv_d, sv_p))

            loss_s = binary_cross_entropy(output_s1, slabels)[1] + binary_cross_entropy(output_s2, slabels)[1]
            
            loss_s.backward()
            self.optG.step()
            self.optC1.step()
            self.optC2.step()
            self.reset_grad()


            # step 2
            output_s1 = self.C1(self.G(sv_d, sv_p))
            output_s2 = self.C2(self.G(sv_d, sv_p))
            output_t1 = self.C1(self.G(v_d, v_p))
            output_t2 = self.C2(self.G(v_d, v_p))            
            
            loss_s1 = binary_cross_entropy(output_s1, slabels)[1]
            loss_s2 = binary_cross_entropy(output_s2, slabels)[1]
            loss_s = loss_s1 + loss_s2
            loss_dis = (nn.Sigmoid()(output_t1) - nn.Sigmoid()(output_t2)).abs().mean()
            
            loss = loss_s - loss_dis
            
            loss.backward()
            self.optC1.step()
            self.optC2.step()
            self.reset_grad()


            # step 3
            output_t1 = self.C1(self.G(v_d, v_p))
            output_t2 = self.C2(self.G(v_d, v_p))
            loss_dis = (nn.Sigmoid()(output_t1) - nn.Sigmoid()(output_t2)).abs().mean()
            loss_dis.backward()
            self.optG.step()
            self.reset_grad()

            loss_epoch[0] += loss_s1.item()
            loss_epoch[1] += loss_s2.item()
            loss_epoch[2] += loss_dis.item()
            

            
            
        loss_epoch[0] = loss_epoch[0] / num_batches
        loss_epoch[1] = loss_epoch[1] / num_batches
        loss_epoch[2] = loss_epoch[2] / num_batches
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch

    def test(self, dataloader="test"):
        test_loss = 0
        y_label, y_pred1, y_pred2 = [], [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            
            self.G.eval()
            self.C1.eval()
            self.C2.eval()
            for i, (v_d, v_p, labels) in enumerate(data_loader):

                v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device)

                score1 = self.C1(self.G(v_d, v_p))
                score2 = self.C2(self.G(v_d, v_p))
                score3 = (score1 + score2) / 2
                
                
                n1, loss1 = binary_cross_entropy(score1, labels)
                n2, loss2 = binary_cross_entropy(score2, labels)
                n3, loss3 = binary_cross_entropy(score3, labels)
                
                test_loss += loss1.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred1 = y_pred1 + n1.to("cpu").tolist()
                y_pred2 = y_pred2 + n2.to("cpu").tolist()
                
                
        auroc = [roc_auc_score(y_label, y_pred1), roc_auc_score(y_label, y_pred2)]
        auprc = [average_precision_score(y_label, y_pred1), average_precision_score(y_label, y_pred2)]
        test_loss = test_loss / num_batches

        return auroc, auprc, test_loss
