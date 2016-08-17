
# coding: utf-8

# # Simple recurrent networks

# In[143]:

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.sparse as sparse
import pprint
get_ipython().magic('matplotlib inline')


# ## Linear network with excitatory and inhibitory neurons

# We start with a network of 100 excitatory and 100 inhibitory neurons:

# In[298]:

REC_EXC_NUM = 100
REC_INH_NUM = 100
N = REC_EXC_NUM + REC_INH_NUM

#Time of the simulation
STEP_DIM = 0.0001
TIME_CONST = 0.01
TOT_TIME = 2
N_STEPS = round(TOT_TIME/STEPS_DIM)
time_vect = np.linspace(0,TOT_TIME, N_STEPS)

#A fix imput to set the firing rates at H0
H_0 = 1

rec_nn = np.random.random((N, N_STEPS))


# To make sure that the system do not diverge, for the excitatory and inhibitory part of the weights matrix we draw values from a gamma distribution. In order to fix the radius (R) of the eigenvalues distribution in the complex plane (we want it lower than 1 for the system not to diverge) we set a theta (scale) of the gamma distribution to make the variance R/sqrt(N)

# In[211]:

R = 0.8 #Limit radius for eigenvalues in complex space

K = 2 #shape of the Gamma distribution
THETA = R/((2*N)**(1/2.0))

w_rec = np.random.gamma(K, THETA, [REC_EXC_NUM+REC_INH_NUM,REC_EXC_NUM+REC_INH_NUM])
np.fill_diagonal(w_rec,0)
scal = sum(w_rec[:,:REC_EXC_NUM].T,1)/sum(w_rec[:,REC_EXC_NUM:].T,1)
w_rec[:,REC_EXC_NUM:] = w_rec[:,REC_EXC_NUM:] * scal.reshape((-1,1))
w_rec[:,REC_EXC_NUM:] = -w_rec[:,REC_EXC_NUM:] 

#Compute eigenvalues:
eig = np.linalg.eigvals(w_rec[:,:])

plt.scatter(eig.real,eig.imag)
plt.axis('equal')
plt.show()


# Run the network:

# In[212]:

np.array([np.ones(REC_EXC_NUM), -np.ones(REC_INH_NUM)])


# In[213]:

for i in range(1,N_STEPS):
    rec_nn[:,i] = rec_nn[:,i-1] + (STEP_DIM/TIME_CONST)*(
                        -rec_nn[:,i-1] + w_rec.dot(rec_nn[:,i-1]) + H_0)


# Since the eigenvalues of the weight matrix are all lower than 1 the network will never diverge:

# In[214]:

for i in range(N):
    plt.plot(time_vect, rec_nn[i,:])
plt.xlim(0,1)
plt.show()


# ###  Stimulus

# We can use a OU process for adding a stimulus to the same network. We start with a spatially uncorrelated (sigma_noise identity matrix) input:

# In[ ]:

sigma_noise


# In[221]:

ell_noise


# In[262]:

sigma_noise = 0.2*np.eye(N)
ell_noise = np.linalg.cholesky(sigma_noise)
tau_noise = 0.05
z_noise = np.sqrt(2*STEP_DIM/tau_noise)

noise = np.zeros(rec_nn.shape)
noise[:,0] = ell_noise.dot(np.random.normal(0,1,(N)))

for i in range(1,N_STEPS):
    noise[:,i] = (1 - STEP_DIM/tau_noise)*noise[:,i-1] + z_noise*ell_noise.dot(np.random.normal(0,1,N))
    rec_nn[:,i] = rec_nn[:,i-1] + (STEP_DIM/TIME_CONST)*(
                        -rec_nn[:,i-1] + w_rec.dot(rec_nn[:,i-1]) + H_0 + noise[:,i-1])


# In[263]:

f,ax = plt.subplots(1,2, figsize = (20,5))
#for i in range(N):
ax[0].imshow(rec_nn[:,:], aspect='auto')
#f.colorbar(ax[0])
ax[1].imshow(noise[:,:], aspect='auto')
#ax[1].plot(time_vect, np.mean(noise,0),)
#ax[1].set_ylim(-5,5)
plt.show()


# Give correlated input to the excitatory and inhibitory population:

# In[325]:

np.mean(a,1)


# In[327]:

a = np.array([[1,2,3],[2,4,3],[3,2,1]])
a/np.tile(np.mean(a,1),(3,1))


# In[350]:

ell_noise = sparse.block_diag([np.ones((round(REC_EXC_NUM/2), round(REC_EXC_NUM/2))), 
                                    np.ones((round(REC_EXC_NUM/2), round(REC_EXC_NUM/2))), 
                                    np.ones((round(REC_INH_NUM/2), round(REC_INH_NUM/2))),
                                    np.ones((round(REC_INH_NUM/2), round(REC_INH_NUM/2)))]).toarray()
ell_noise = ell_noise/np.tile(np.mean(ell_noise,1),(ell_noise.shape[0],1))
#sigma_noise = 0.2*np.eye(N)
#ell_noise = np.linalg.cholesky(sigma_noise)
tau_noise = 0.5
z_noise = np.sqrt(2*STEP_DIM/tau_noise)

noise = np.zeros(rec_nn.shape)
noise[:,0] = ell_noise.dot(np.random.normal(0,1,(N)))

for i in range(1,N_STEPS):
    noise[:,i] = (1 - STEP_DIM/tau_noise)*noise[:,i-1] + z_noise*ell_noise.dot(np.random.normal(0,1,N))
    rec_nn[:,i] = rec_nn[:,i-1] + (STEP_DIM/TIME_CONST)*(
                        -rec_nn[:,i-1] + w_rec.dot(rec_nn[:,i-1]) + H_0 + noise[:,i-1])


# In[351]:

f,ax = plt.subplots(2,2, figsize = (20,10))
for i in range(N):
    ax[0,0].plot(time_vect,rec_nn[i,:])
    #ax[0,1].plot(time_vect, noise[i,:])
    
ax[1,0].imshow(rec_nn[:,:], aspect='auto')
#f.colorbar(ax[0])
ax[1,1].imshow(noise[:,:], aspect='auto')
#ax[1].plot(time_vect, np.mean(noise,0),)
#ax[1].set_ylim(-5,5)
plt.show()


# Correlated input only to a subpopulation:

# In[340]:

np.mean(ell_noise,0)


# In[362]:

#ell_noise = sparse.block_diag([np.ones((round(REC_EXC_NUM/2), round(REC_EXC_NUM/2))), 
#                                    np.ones((round(REC_EXC_NUM/2), round(REC_EXC_NUM/2))), 
#                                    np.ones((round(REC_INH_NUM/2), round(REC_INH_NUM/2))),
#                                    np.ones((round(REC_INH_NUM/2), round(REC_INH_NUM/2)))]).toarray()
N_CORR = 20

sigma_noise = 0.2*np.eye(N)
#ell_noise = np.linalg.cholesky(sigma_noise)
#ell_noise = sigma_noise
sigma_noise[:N_CORR,:N_CORR] = sigma_noise[0,0]
ell_noise = sigma_noise

ell_noise = ell_noise/np.tile(np.sum(ell_noise,1),(ell_noise.shape[0],1))

tau_noise = 0.05
z_noise = np.sqrt(2*STEP_DIM/tau_noise)

noise = np.zeros(rec_nn.shape)
noise[:,0] = ell_noise.dot(np.random.normal(0,1,(N)))

for i in range(1,N_STEPS):
    noise[:,i] = (1 - STEP_DIM/tau_noise)*noise[:,i-1] + z_noise*ell_noise.dot(np.random.normal(0,1,N))
    rec_nn[:,i] = rec_nn[:,i-1] + (STEP_DIM/TIME_CONST)*(
                        -rec_nn[:,i-1] + w_rec.dot(rec_nn[:,i-1]) + noise[:,i-1])


# In[363]:

f,ax = plt.subplots(2,2, figsize = (20,10))
for i in range(N):
    ax[0,0].plot(time_vect,rec_nn[i,:],'b')

ax[0,0].plot(time_vect, np.mean(noise,0)/np.max(np.mean(noise,0)), 'r', linewidth=2)
ax[0,0].plot(time_vect, np.mean(noise[:N_CORR],0)/np.max(np.mean(noise[:N_CORR],0)), 'k', linewidth=2)

ax[0,1].plot(time_vect, noise[0,:])
ax[0,1].plot(time_vect, noise[50,:])
    
ax[1,0].imshow(rec_nn[:,:], aspect='auto')
#f.colorbar(ax[0])
ax[1,1].imshow(noise[:,:], aspect='auto')
#ax[1].plot(time_vect, np.mean(noise,0),)
#ax[1].set_ylim(-5,5)
plt.show()


# In[ ]:




# In[ ]:



