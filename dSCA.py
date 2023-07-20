# read data
import dill
import numpy as np
import tensorflow as tf


def Learn(X,X2=None,test=False,n_components=3,alpha=0.5,n_iter=600,lossname='lossEla'):
    
    X=tf.Variable(X)
    if test:
        X2=tf.Variable(X2)
    
    def lossL2(): 
        los=0.5*tf.norm(X-tf.matmul(U,V),ord=2)**2+tf.math.scalar_mul(alpha,tf.norm(V,ord=2))**2 
        return(los) 

    def lossL1(): 
        los=0.5*tf.norm(X-tf.matmul(U,V),ord=2)**2+tf.math.scalar_mul(alpha,tf.norm(V,ord=1)) 
        return(los) 

    def lossEla(): 
        los=0.5*tf.norm(X-tf.matmul(U,V),ord=2)**2+tf.math.scalar_mul(alpha,tf.norm(V,ord=1))+tf.math.scalar_mul(0.05*alpha,tf.norm(V,ord=2))**2 
        return(los) 

    def err():
        los=0.5*tf.norm(X2-tf.matmul(U,V),ord=2)**2
        return(los)
    
    lossfunc=eval(lossname)
    
    alpha=tf.constant(alpha,dtype='float64')
                                    
    V=tf.Variable(np.random.randn(n_components,X.shape[1]))
    opt = tf.keras.optimizers.Adam(learning_rate=0.1,)
    
    losses=[]
    errs=[]
    for i in range(n_iter):

        #update U
        U=tf.transpose(tf.linalg.lstsq(tf.transpose(V)\
            ,tf.transpose(X),0.001))
        U=tf.linalg.normalize(U,axis=0)[0]

        #update V
        opt.minimize(lossfunc,var_list=[V])
        
        
        
        #log training and validation loss
        losses.append(lossfunc().numpy())
        if test:
            errs.append(err().numpy())
        
        if i>80:

            if losses[i]>=np.array(losses[(i-5):i]).mean():
                break

    U=tf.Variable(U.numpy(),\
                  constraint=tf.keras.constraints.UnitNorm(axis=0)) 

    LRschedule=tf.keras.optimizers.schedules.ExponentialDecay(\
        0.01, 600, 0.15, staircase=False, name=None)
    opt = tf.keras.optimizers.SGD(learning_rate=LRschedule)
    
    n_iter2=n_iter
    for i in range(n_iter2):


        opt.minimize(lossfunc,var_list=[U,V])

        #log training and validation loss
        losses.append(lossfunc().numpy())
        if test:
            errs.append(err().numpy())
        
        #early stopping
        if i>80:

            if losses[-1]>=np.array(losses[-6:-1]).mean():
                break
    
    res=[U.numpy(),V.numpy(),losses,errs]
    return(res)



        



