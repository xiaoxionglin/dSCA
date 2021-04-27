import dSCA
import dill
from datetime import datetime
import os
import numpy as np

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

# this folder is where you have put your data and where you will store the grid search results
folder='demo'

#these are specific to my analysis, which you can just leave here or change it to your liking

#the optimization method, here I use ADMM followed by normal gradient descent. It's just a name for my log file.
version='MixOpt'
type_restructure='s100s50'
stitching='SDT'

#here you may want to iterate through all the stimuli and there interaction
for Stim in ['ST']:#['DT+SDT']:
    
    #here I wanted to compare L1 and L2 regularizer, you could just use L1
    for regu in ['L1']:#,'L2']:
        
        #this the data filename
        filename=folder+'/splitX.pkl'
        #+stitching+'_'\
        
        #here my data contains already the training and test set.
        #you could just import the raw data and split them into train and test set here
        with open(filename,'rb') as f:
            tmpX1=dill.load(f)
            tmpX2=dill.load(f)
    
        # the name for log file
        name='GS_'+Stim+'_'+stitching+'_'+regu+'_'+type_restructure+'_'+version;
        
        if regu=='L1':
            lossname='lossEla'
        else:
            lossname='lossL2'
        
        #number of runs.      if the data is well-structured, you should change this loop into repeat&fold by e.g. using the method in sklearn
        n_run=1#4
        
        max_components=3#11
        
        #how fine grained you want to explore the different alpha values in [10^alpha_s,10^alpha_e]
        n_alphas=3#80
        alpha_s=-1
        alpha_e=0.5
        
        #log training loss
        lossall=np.zeros((n_run,max_components,n_alphas));
        valossall=np.zeros((n_run,max_components,n_alphas));
        
        #log testing loss
        errall=np.zeros((n_run,max_components,n_alphas));

        
        



        os.mkdir(folder+'/ckpt_'+name)
        os.mkdir(folder+'/ckpt_'+name+'/weights')
        
        for n_run_i in range(n_run):

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logfile=open(folder+'/logfile_'+name+'.txt','a')
            logfile.write('n_run %d/%d   \t\t\t\t\t %s\n'%(n_run_i,n_run,current_time));
            
            #here I switched train and test set, which may not suit you (you may have more than 2 folds)
            tmpX1,tmpX2=tmpX2,tmpX1

            for j,n_components in zip(range(0,max_components),range(1,max_components+1)):
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logfile.write('\t component %d/%d\t\t\t\t\t %s\n'%(n_components,max_components,current_time));
                for k,al in zip(range(n_alphas),np.logspace(alpha_s,alpha_e,n_alphas)):
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    logfile.write('\t \t alpha %d/%d\t\t\t\t\t %s\n'%(k,n_alphas,current_time));
                    
                    
                    U,V,losses,errs=dSCA.Learn(tmpX1,tmpX2,True,n_components,al,lossname=lossname)
                        
                    #log training and validation loss
                    lossall[n_run_i,j,k]=losses[-1]
                    errall[n_run_i,j,k]=errs[-1]
                    
                    
                    
                    with open(folder+'/ckpt_'+name+'/weights/nn%02dweights%02d%02d.pkl'%(n_run_i,j,k),'wb') as f:
                        dill.dump(U,f)
                        dill.dump(V,f)
                        dill.dump(losses,f)
                        dill.dump(errs,f)

            logfile.write('\n')
            logfile.close()


        with open(folder+'/ckpt_'+name+'/losses_TrVa.pkl','wb') as f:
            dill.dump(lossall,f)
            dill.dump(errall,f)