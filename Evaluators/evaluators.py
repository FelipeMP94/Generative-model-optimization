import numpy as np
import torch
from torch import nn

class evaluator_Best_Mean(nn.Module):
    def __init__(self,decoder):
        super(evaluator_Best_Mean,self).__init__()
        self.is_inicialized = False 
        self.decoder = decoder
        self.best_fit = 0
        self.mean_current_pop= 0
        self.epochs_best_fit = []
        self.epochs_mean_fit = []
        
        
        
    def _loss(self,melhor,media):
        x = (melhor-media)/media
        loss = np.exp(-1*x)
        return loss

    def best_chromossome(self,population):
        ch_fit = []
        for index,chromosome in enumerate(population):
            fit = self.decoder.decode(chromosome,True)
            ch_fit.append((fit,index))
            ch_fit.sort(reverse=True)
        return ch_fit
           

        
        
    def forward(self,population):
        pop_individual_fit = np.array([],dtype=int)
        for index,chromosome in enumerate(population):
            fit = self.decoder.decode(chromosome,True)
            pop_individual_fit = np.append(pop_individual_fit,fit)
        
        if self.is_inicialized == False:
            self.best_fit = np.min(pop_individual_fit)
            self.mean_current_pop = np.mean(pop_individual_fit)
                
            self.epochs_best_fit.append(self.best_fit)
            self.epochs_mean_fit.append(self.mean_current_pop)
            
    
            loss = self._loss(self.best_fit,self.mean_current_pop)
            
            
    
            self.is_inicialized = True
        else:
            self.mean_current_pop = np.mean(pop_individual_fit)
            self.epochs_mean_fit.append(self.mean_current_pop)
     
            
            loss = self._loss(self.best_fit,self.mean_current_pop)
            
            best_individual_fit = np.min(pop_individual_fit)
                          
            if best_individual_fit < self.best_fit:
                self.best_fit = best_individual_fit
                
                 
        return torch.tensor([loss])