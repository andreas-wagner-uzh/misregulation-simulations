# -*- coding: utf-8 -*-
"""
simulates evolution of misregulation under selection-mutation-drift dynamics,
with mutation rates estimated from mouse TFBSs
"""

#needed to load mutation rate data
import pandas as pd

# for random number generators
import random 

import numpy as np

#functions needed for population simulation
import popfuncs




#to print time stamped data
from datetime import datetime
dateFORMAT = '%d-%m-%Y'



#write to a time stamped output file
dateFORMAT = '%d-%m-%Y'
outfilename="popsim_"+ \
             datetime.now().strftime(dateFORMAT) + "_out.txt"
print("\nwriting output to ", outfilename)


outfile = open(outfilename, 'w')
#print a header for the output file
#fGspace corresponds to pB, the fraction of sequence space filled with TFBSs
#prefixes 'm' refer to means, prefixes 's' to a measure of scatter
print("L\tgenctr\tmu\tN\tG\tGon\ts01\ts10\tfGspace", end ='', file = outfile)  
print("\tmu0p\tmu0m\tmu1p\tmu1m", end ='', file = outfile)  
print("\tmfit\tsfit\tmf00\tsf00\tmf01\tsf01\tmf10\tsf10\tmf11\tsf11\tmdeltam\tsdeltam", end ='', file = outfile)
outfile.close()


############################################################
## first load data on mutation probabilities
## this data contains the likelihoods that a mutation
## creates or destroys transcription factor binding sites (TFBSs). 
## It is based on mouse protein binding microarray data for 
## multiple mouse transcription factors, as described in the
## methods section of the paper.
#############################################################

mutdatfile="mutation_data.txt"
df =  pd.read_csv(mutdatfile, sep='\t')
#convert data frame to dictionary for easier handling
mutdatall = df.to_dict()

#pB is the fraction of sequence space filled with TFBSs
#pBlist contains a list of pB values for which mutation
#probabilities have been calculated by simulation
pBlist = sorted(list(mutdatall["nBSs"].keys()));

#dictionaries keyed by pB that will hold the mutation likelihood data
#The simulations in here use mu0p=mu1p and 
#mu0m=mu1m, so the loading of different probabilities is a legacy 
#of earlier versions of this code
#the likelihood of creating a TFBS for a gene that should be off
mu0p_d={}
#the likelihood of destroying a TFBS for a gene that should be off
mu0m_d={}
#the likelihood of creating a TFBS for a gene that should be on
mu1p_d={}
#the likelihood of destroying a TFBS for a gene that should be on
mu1m_d={}

#load the mutation probability data that is needed here
for pBtmp in pBlist:
    mu0p_d[pBtmp]=mutdatall["s_mum_bio"][pBtmp]
    mu0m_d[pBtmp]=mutdatall["r"][pBtmp]
    mu1p_d[pBtmp]=mutdatall["s_mup_bio"][pBtmp]
    mu1m_d[pBtmp]=mu0m_d[pBtmp]

#print console output just to make sure that we have loaded the right data
print("\n\nmutation rate table\npB\tmu0p\tmu0m\tmu1p\tmu1m")
for pBtmp in pBlist:
    print(pBtmp, mu0p_d[pBtmp],mu0m_d[pBtmp],mu1p_d[pBtmp],mu1m_d[pBtmp])


#transfer the same data to arrays, which are needed for the interpolation below  
pBarr=np.array(sorted(mu1m_d.keys()))
mu0marr=[]
mu0parr=[]
mu1marr=[]
mu1parr=[]
for pBtmp in pBarr:
    #the probability that an active binding site gets destroyed, estimated
    #from PBM data
    mu0marr.append(mu0m_d[pBtmp])
    mu0parr.append(mu0p_d[pBtmp])
    mu1marr.append(mu1m_d[pBtmp])
    mu1parr.append(mu1p_d[pBtmp])
mu0marr=np.array(mu0marr) 
mu0parr=np.array(mu0parr) 
mu1marr=np.array(mu1marr) 
mu1parr=np.array(mu1parr)   

# interpolations of mutation likelihoods for values of pB different from those
# that were simulated
# useful to compute mutation rates at specific pB values    
from scipy.interpolate import interp1d
mu0p_i = interp1d(pBarr, mu0parr, kind='cubic')
mu0m_i = interp1d(pBarr, mu0marr, kind='cubic')
mu1p_i = interp1d(pBarr, mu1parr, kind='cubic')
mu1m_i = interp1d(pBarr, mu1marr, kind='cubic')


#Length of TFBSs, needed to compute mutation rates from mutation likelihoods below
L=8

#mutation rate per nucleotide, 
#an array with only one value is used
#in the loop below 
for mu in [1e-5]:
    for popsize in [1e3]:
        popsize=int(popsize)
        for G in [1500]:
            for Gon in [G/2, 3*G/4, G/4]:
                #scale selection coefficients by population size
                for s01 in [0.01/popsize, 0.1/popsize, 1/popsize, 10/popsize, 100/popsize]:
                    for s10 in [0.01/popsize, 0.1/popsize, 1/popsize, 10/popsize, 100/popsize]:
                        #the fraction of genotype space filled with TFBSs
                        for pB in [0.05, 0.15, 0.25, 0.35, 0.45]:
                            
                            #muxy_i is the likelihood that a single mutation 
                            #creates or destroys a TFBSs. These likelihoods are
                            # here transformed into actual mutation rates needed
                            #for population simulation
                            mu0p=L*mu*mu0p_i(pB)
                            mu0m=L*mu*mu0m_i(pB)
                            mu1p=L*mu*mu1p_i(pB)
                            mu1m=L*mu*mu1m_i(pB)                         
                            
                          
                            #initialize a population
                            #each individual is a 1D numpy array
                            #entry 0 (binary 00): binding sites 00 (inactive and supposed to be inactive) 
                            #entry 1 (binary 01): binding sites 01 (active and supposed to be inactive) 
                            #entry 2 (binary 10): binding sites 10 (inactive and supposed to be active) 
                            #entry 3 (binary 11): binding sites 11 (active and supposed to be active) 
                            #entry 4: fitness
                            #the whole population is a matrix of these individuals
                            pop=popfuncs.popinit_bin_noCT_nparr_simple(popsize, G, Gon, fit=1)
                          
                            #the number of generations to simulate
                            maxgen=int(1/mu)
                            #determines the generation indices at which we report output
                            plotgen=int((maxgen-1)/20)
                            
                            #number of generations to average over for output
                            #averaging will only work if the time intervals for printing output
                            #are much longer than the time intervals for averaging. 
                            #No exception handling is implemented to enforce this condition  
                            gen_ave=100 
                            #a flag to indicate below whether we are currently in a time window
                            #where we collect data for output
                            ave_ctr=-1
                            
                            #core of the population simulation
                            for genctr in range(maxgen):
                                 
                                #mutate and calculate fitness post-mutation
                                #note that mu0p=mu1p and mu0m=mu1m, because we
                                #do not make a difference between TFBSs of genes
                                #that should be on or off with respect to their
                                #mutation probability
                                popfuncs.mutpop_calc_fit_simple(pop, G, Gon, mu0p, mu0m, mu0p, mu0m, s01, s10)
                                pop=popfuncs.select_soft_nparr_simple(pop)
                                
                                #begin data collection in a generation that is a multiple of plotgen
                                #and continue for gen_ave generations
                                if(genctr/plotgen==int(genctr/plotgen)):
                                    ave_ctr=0
                                    #reset arrays that hold statistics to be averaged over
                                    mfit=np.zeros(gen_ave)
                                    mf00=np.zeros(gen_ave)
                                    mf01=np.zeros(gen_ave)
                                    mf10=np.zeros(gen_ave)
                                    mf11=np.zeros(gen_ave)
                                    mdeltam=np.zeros(gen_ave)                          
                            
                                    
                                if(ave_ctr>-1 and ave_ctr<gen_ave): 
                                    
                                    #calculate mean and sdev of population fitness       
                                    fitstats=popfuncs.calc_fitstats_simple(pop)
                                    
                                    #calculate mean and sdev of the fraction of TFBSs in each
                                    #category of TFBS over the whole population
                                    #The categories are
                                    #binding sites 00 (inactive and supposed to be inactive) 
                                    #binding sites 01 (active and supposed to be inactive) 
                                    #binding sites 10 (inactive and supposed to be active) 
                                    #binding sites 11 (active and supposed to be active) 
                                    
                                    BSstats=popfuncs.calc_BSstats_bin_simple(pop,  0, G, Gon)
                                   
                                        
                                    mfit[ave_ctr]=fitstats[0]
                                    mf00[ave_ctr]=BSstats[0][0][0]
                                    mf01[ave_ctr]=BSstats[0][0][1]
                                    mf10[ave_ctr]=BSstats[0][1][0]
                                    mf11[ave_ctr]=BSstats[0][1][1]
                                    mdeltam[ave_ctr]=(BSstats[0][0][1]*((G-Gon)/G )) - (BSstats[0][1][0]*(Gon / G) ) 
                                    
                                    ave_ctr=ave_ctr+1
                                    
                                #if we have averaged over gen_aveenough generations, output the averages 
                                #and reset the flag for averaging
                                if(ave_ctr==gen_ave):
                                    
                                    #for monitoring simulation progress on the console:
                                    print('\nprogress {0:5d}\t'.format(genctr), end ='')
                                    print('{0:.1e}\t'.format(mu), end ='')
                                    print('{0:.1e}\t'.format(popsize), end ='')
                                    print('{0:5d}\t'.format(G), end ='')
                                    print('{0:5d}\t'.format(int(Gon)), end ='')
                                    print('{0:.2e}\t'.format(s01), end ='')
                                    print('{0:.2e}\t'.format(s10), end ='')
                                    print('{0:.5f}\t'.format(pB), end ='')
                                    
                                    #open outfile for appending
                                    outfile = open(outfilename, 'a')
                                    
                                    print('\n{0:1d}\t'.format(L), end ='', file = outfile)
                                    print('{0:5d}\t'.format(genctr), end ='', file = outfile)
                                    print('{0:.1e}\t'.format(mu), end ='', file = outfile)
                                    print('{0:.1e}\t'.format(popsize), end ='', file = outfile)
                                    print('{0:5d}\t'.format(G), end ='', file = outfile)
                                    print('{0:5d}\t'.format(int(Gon)), end ='', file = outfile)
                                    print('{0:.2e}\t'.format(s01), end ='', file = outfile)
                                    print('{0:.2e}\t'.format(s10), end ='', file = outfile)
                                    print('{0:.5f}\t'.format(pB), end ='', file = outfile)
                                    
                                    print('{0:.2e}\t'.format(mu0p), end ='', file = outfile)
                                    print('{0:.2e}\t'.format(mu0m), end ='', file = outfile)
                                    print('{0:.2e}\t'.format(mu1p), end ='', file = outfile)
                                    print('{0:.2e}\t'.format(mu1m), end ='', file = outfile)
                                    
                                    print('{0:.4f}\t{1:.4f}\t'.format(np.mean(mfit), np.std(mfit)), end ='', file = outfile)
                                    print('{0:.2e}\t{1:.2e}\t'.format(np.mean(mf00), np.std(mf00)), end ='', file = outfile)
                                    print('{0:.2e}\t{1:.2e}\t'.format(np.mean(mf01), np.std(mf01)), end ='', file = outfile)
                                    print('{0:.2e}\t{1:.2e}\t'.format(np.mean(mf10), np.std(mf10)), end ='', file = outfile)
                                    print('{0:.2e}\t{1:.2e}\t'.format(np.mean(mf11), np.std(mf11)), end ='', file = outfile)
                                    
                                    print('{0:.2e}\t{1:.2e}\t'.format(np.mean(mdeltam), np.std(mdeltam)), end ='', file = outfile)
                                    
                                     
                                    outfile.close()
                                    
                                    #reset the flag for averaging
                                    ave_ctr=-1
                                
                                
                           
       
        
        

    



    
    
 