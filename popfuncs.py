# -*- coding: utf-8 -*-

"""functions necessary for population dynamic simulations"""

import random # for random number generators
import numpy as np #for poisson distribution and vectorization
import copy #to make deep copies of nested data structures

#the multiplicative fitness expression
def mfit_mult(G, Gon, f01, f10, s01, s10):
    return ((1-s01)**((G-Gon)*f01))*((1-s10)**((Gon)*f10)) 
   
#initialize a population
#each individual is a 1D numpy array
#entry 0 (binary 00): binding sites 00 (inactive and supposed to be inactive) 
#entry 1 (binary 01): binding sites 01 (active and supposed to be inactive) 
#entry 2 (binary 10): binding sites 10 (inactive and supposed to be active) 
#entry 3 (binary 11): binding sites 11 (active and supposed to be active) 
#entry 4: fitness
#the whole population is a matrix of these individuals
def popinit_bin_noCT_nparr_simple(popsize, G, Gon, fit):
    """
    create a population of size popsize, where there is no misregulation, 
    that is, all genes that are supposed to 
    be on are on, and all those that are supposed to be off are off
    fitness of each individual is set to one """
    #make sure that fitness is a float, all others can be integers
    pop = [ [ G-Gon, 0, 0, Gon, float(fit)]  for _ in range(popsize)]
    return np.array(pop)


#mutate members of a population and calculate their fitness 
def mutpop_calc_fit_simple(pop, G, Gon, mu0p, mu0m, mu1p, mu1m, s01, s10):
    """mutate binding sites for individuals in an entire population
    where the population has a simple data structure,
    a numpy array with dimension popsize times 5
    will only be efficient of many individuals in a mutation experience mutations"""
    
    popsize = len(pop)
    
    #first mutate TFBSs of genes that should be off
 
    #now create a vector that will hold the expected number of mutational changes for 
    #binding sites supposed to be inactive
    #first for binding sites that are inactive, where mutations can create new BSs
    meancreate=[ mu0p*pop[i, 0] for i in range(popsize) ]
    #and then for BSs that are active,which mutations can destroy
    meandestroy=[ mu0m*pop[i, 1] for i in range(popsize) ]
    #now create poisson random variates for each of these means
    nmutcreate=np.random.poisson(lam=meancreate, size=popsize)
    nmutdestroy=np.random.poisson(lam=meandestroy, size=popsize)
    
    #the change in the number of active TFBSs
    deltan = nmutcreate - nmutdestroy
    
     
    for i in range(popsize): 
        if deltan[i]==0:
            continue
        #if more active binding sites are to be destroyed than actually exist, set the 
        #new number of active binding sites to the minimally possible value zero
        elif(pop[i,1] + deltan[i] <0):
            pop[i,1]=0
            #G-Gon is the maximal number of BSs that can be wrongly active
            pop[i,0]=G-Gon
        else:
            #net functional Bss to be created -> will reduce the number of inactive BSs
            pop[i,0] =  pop[i,0] - deltan[i]
            #net functional BSs to be created, will increase the number of active TFBSs
            pop[i,1] =  pop[i,1] + deltan[i] 
   
    #now mutate TFBSs of genes that should be off, using the exact same procedure    
    meancreate=[ mu1p*pop[i, 2] for i in range(popsize) ]
    meandestroy=[ mu1m*pop[i, 3] for i in range(popsize) ]
    nmutcreate=np.random.poisson(lam=meancreate, size=popsize)
    nmutdestroy=np.random.poisson(lam=meandestroy, size=popsize)      
    deltan = nmutcreate - nmutdestroy
    
    for i in range(popsize):
        if deltan[i]==0:
            continue
        elif(pop[i,3] + deltan[i] <0):
            pop[i,3]=0
            pop[i,2]=Gon
        else:
            pop[i,2] =  pop[i,2] - deltan[i]
            pop[i,3] =  pop[i,3] + deltan[i] 
   
    #after mutation, calculate the fitness of each individual
    for i in range(popsize):
        pop[i,4]=((1-s01)**(pop[i,1]))*((1-s10)**(pop[i,2]))
    
    


#uses vectorization through numpy
def select_soft_nparr_simple(oldpop):
    """Implement soft selection. Assumes that fitness values of
    individuals lie between 0 and 1, no exception handling for the sake of speed"""
    my_popsize=len(oldpop)
    #write fitness values into numpy array
    fitlist=np.array([ind[4] for ind in oldpop]) 
    #normalization necessary because choice function below 
    #needs fitness values to be probabilities
    fitlist = fitlist/np.sum(fitlist) 
     
    #note that the first argument of choice is an array that just runs through the index of individuals
    newpopref=np.random.choice([x for x in range(my_popsize)], size=my_popsize, replace=True, p=fitlist)     
    #big but unavoidable time sink of deep copying
    newpop = copy.deepcopy(np.array([oldpop[newpopref[i]] for i in range(my_popsize)])) 
    
    return newpop  

#calculate mean and sdev of population fitness
def calc_fitstats_simple(pop):
    fitarr=[ pop[i,4] for i in range(0,len(pop)) ]
    return [ np.mean(fitarr), np.std(fitarr) ]  

#counts the fraction of TFBSs in each of the four categories (00, 01, 10, 11)
#the returned array arr[0] will be a 2x2 array holding the means, and 
#arr[1] will be a 2x2 array holding the standard deviations of these fractions for
#each of the four categories of TFBSs, where mean and sdev
#are taken over the whole population
def calc_BSstats_bin_simple(pop, TFindex, G, Gon):
   
    #create a numpy array to allow division by an integer
    npromG00=np.array([ pop[i,0] for i in range(0,len(pop)) ])
    #divide to calculate the fraction of BSs
    npromG00 = npromG00/(G-Gon)
    #proceed analogously for the other TFBS categories  
    npromG01=np.array([ pop[i,1] for i in range(0,len(pop)) ])
    npromG01 = npromG01/(G-Gon)
    npromG10=np.array([ pop[i,2] for i in range(0,len(pop)) ])
    npromG10 = npromG10/Gon
    npromG11=np.array([ pop[i,3] for i in range(0,len(pop)) ])
    npromG11 = npromG11/Gon
    #the returned array arr[0] will be a 2x2 array holding the means, and 
    #arr[1] will be a 2x2 array holding the standard deviations
    return [ [[np.mean(npromG00), np.mean(npromG01)], [np.mean(npromG10), np.mean(npromG11)]], \
             [[np.std(npromG00), np.std(npromG01)], [np.std(npromG10), np.std(npromG11)]] ]
               
    
  





    
    
    
