
# coding: utf-8

# # BCM Rule - recurrent network

# ## Single postsynaptic neuron

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
get_ipython().magic('matplotlib inline')


# Updating functions for a single neuron $y$, its weights $w$ and the threshold $\theta$:
# $$ \mathbf{y}(t) = \mathbf{w}\mathbf{y} $$
# $$ \frac{d\mathbf{w}}{dt} = \eta \mathbf{x} \mathbf{y}(\mathbf{y}(t)-\mathbf{\theta}) $$
# $$ \frac{d\mathbf{\theta}}{dt} = \frac{\mathbf{y}^2}{y_0} $$

# In[ ]:

class NeuralNet(object):
    
    def __init__(self, FF_NUM=10, REC_NUM=10, TOT_TIME=20, STEPS_DIM=0.0001, STARTING_RATE = 20,
                   TARGET_RATE=10, THETA_START=0.07, ETA=float('1e-2'), TAU=0.03, TIME_CONST = 1,
                   FF_PL=False, REC_PL=False):
        self.FF_NUM = FF_NUM
        self.REC_NUM = REC_NUM
        self.TOT_TIME = TOT_TIME
        self.STEPS_DIM = STEPS_DIM
        self.STARTING_RATE = STARTING_RATE #starting rate for recurrent network
        self.TARGET_RATE = TARGET_RATE
        self.THETA_START = THETA_START
        self.ETA = ETA
        self.TAU = TAU #for weights
        self.FF_PL = FF_PL
        self.REC_PL = REC_PL
        self.TIME_CONST = TIME_CONST #for rate
        
        self.change_values()
        self.set_stimulation()
        
        
    def __repr__(self):
        return '''%d FF neurons\n%d REC neurons''' % (self.FF_NUM, self.REC_NUM) 
        
        
    def change_values(self):
        
        self.STEPS_N = round(self.TOT_TIME/self.STEPS_DIM) # time steps = total simulation time / simulation step size
        self.time_vect = np.linspace(0,self.TOT_TIME, self.STEPS_N) # create a time vector of size TOT.time and step size 
        
        self.rec_nn = np.zeros((self.REC_NUM, self.STEPS_N)) # create empty array with size of the recurrent network
        self.rec_nn[:,0] = np.random.randint(0,self.STARTING_RATE,self.REC_NUM) # intial firing rates at time step 0
        
        self.ff_nn = np.zeros((self.REC_NUM, self.STEPS_N))
        
        # create a random weight matrix for feed-forward input (size REC x FF) 
        self.w_ff = np.random.uniform(0.5, 2, [self.REC_NUM, self.FF_NUM]) #random uniform rates [0.5, 2] Hz (Clopath)
        
        # create a random weight matrix for the recurrent network
        self.w_rec = np.zeros((self.REC_NUM, self.REC_NUM, self.STEPS_N))
        self.w_rec[:,:,0] = 0.25 #recurrent weights of 0.25 Hz (Clopath)
        
        self.theta = np.zeros((self.REC_NUM, self.STEPS_N))
        self.theta[:,0] = self.THETA_START
    
    
    def set_stimulation(self, BACKGROUND_TYPE='noisy', STIM_TYPE='square_wave', 
                        STIM_VALUE=None, NOISE_VALUE=None, STIM_LENGTH=None, STIM_END=None,
                        target_neurons=None):
                        #target_neurons=[]):
        if STIM_VALUE is None:
            STIM_VALUE = self.STARTING_RATE/4
        if NOISE_VALUE is None:
            NOISE_VALUE = STIM_VALUE/4
        if STIM_END is None:
            STIM_END = self.TOT_TIME
        if STIM_LENGTH is None:
            STIM_LENGTH = round(self.TOT_TIME/2)
        if target_neurons is None:
            target_neurons = list(range(round(self.REC_NUM/5)))
        
        #Set the background (input values to nonstimulated neurons - zero or random noise)
        if BACKGROUND_TYPE=='none':
            #Set the input values of the non stimulated neurons
            self.ff_nn = np.zeros((self.REC_NUM, self.STEPS_N)) 
            #Set the gaps of the periodic signal without stimulus
            periodic_sig = np.zeros((len(target_neurons), round(STIM_LENGTH*2/self.STEPS_DIM)))
            
        elif BACKGROUND_TYPE=='noisy':
            #Set the input values of the non stimulated neurons
            self.ff_nn = np.random.rand(
                self.REC_NUM, self.time_vect.shape[0])*NOISE_VALUE 
            #Set the gaps of the periodic signal without stimulus
            periodic_sig = np.random.rand(
                len(target_neurons), round(STIM_LENGTH*2/self.STEPS_DIM))*NOISE_VALUE
        
        #Set the stimulation values
        if STIM_TYPE=='square_wave':
            #Non-noisy square wave:
            periodic_sig[:,0:round(STIM_LENGTH/self.STEPS_DIM)] = STIM_VALUE
        elif STIM_TYPE=='noisy_square_wave':
            #Noisy square wave:
            periodic_sig[:,0:round(STIM_LENGTH/self.STEPS_DIM)] = STIM_VALUE + np.random.rand(
                len(target_neurons), round(STIM_LENGTH/self.STEPS_DIM))*NOISE_VALUE
        
        #Multiple stimuli if stim_length is shorter than total time: 
        stim_template = np.tile(periodic_sig, [1, round(STIM_END/STIM_LENGTH)+1])
        #print(self.ff_nn.shape)
        self.ff_nn[target_neurons,:round(STIM_END/self.STEPS_DIM)] = stim_template[
            :,:round(STIM_END/self.STEPS_DIM)]

    def display_stim(self):
        f, ax = plt.subplots(1,1, figsize = (20,5))
        ax.imshow(self.ff_nn, aspect='auto',  interpolation='nearest')
    
    def run_network(self):
        for i in range(1,self.STEPS_N):
            np.fill_diagonal(self.w_rec[:,:,i-1], 0) # set diagonal to 0 to prevent self-excitation
            
            self.rec_nn[:,i] = self.rec_nn[:,i-1] + (self.STEPS_DIM/self.TIME_CONST)*(
                    -self.rec_nn[:,i-1] + self.w_rec[:,:,i-1].dot(self.rec_nn[:,i-1]) + 
                    self.w_ff[:,:].dot(self.ff_nn[:,i-1]))
            
            self.theta[:,i] = self.theta[:,i-1] + (self.STEPS_DIM/self.TAU) * (
                -self.theta[:,i-1] + (self.rec_nn[:,i]**2)/self.TARGET_RATE)
                        
            if self.REC_PL:
                self.w_rec[:,:,i-1] = self.w_rec[:,:,i-1].clip(0)
                
                self.w_rec[:,:,i] = self.w_rec[:,:,i-1] + self.STEPS_DIM*self.ETA *(
                    self.w_rec[:,:,i-1] * (self.rec_nn[:,i-1]*(self.rec_nn[:,i-1] - self.theta[:,i-1])).T).T
                
            elif self.FF_PL:
                self.w_ff[:,:,i-1] = self.w_ff[:,:,i-1].clip(0)
                self.w_ff[:,:] = self.w_ff[:,:] + self.STEPS_DIM*self.ETA *(
                    self.w_ff[:,:] * (self.ff_nn[:,i-1]*(self.ff_nn[:,i-1] - self.theta[:,i-1])).T).T
                
    def plot_nn(self):
        f, ax = plt.subplots(1,2, figsize = (20,5))
        
        ax[0].plot(self.time_vect, [plastic_recurr.TARGET_RATE]*plastic_recurr.time_vect.shape[0], 'k--')
                   
        for i in range(self.REC_NUM):
            ax[0].plot(self.time_vect, self.rec_nn[i,0:self.time_vect.shape[0]])
        
        for i in range(self.FF_NUM):
            ax[0].plot(self.time_vect, self.ff_nn[i,0:self.time_vect.shape[0]], 'k')
            
        rec_weight_map = ax[1].pcolor(self.w_rec[:,:,self.STEPS_N-1], cmap=plt.cm.Blues)
#        f[1].colorbar(rec_weight_map)
        
            


# In[ ]:

#%%timeit
plastic_recurr = NeuralNet()
plastic_recurr.set_stimulation(STIM_LENGTH=1)


# In[ ]:

#%%timeit
plastic_recurr.run_network()


# In[ ]:

plastic_recurr.plot_nn()


# In[ ]:

print(plastic_recurr.w_rec[:,:,plastic_recurr.STEPS_N-1])


# In[ ]:

plastic_recurr.display_stim()


# In[ ]:




# In[ ]:




# In[ ]:



