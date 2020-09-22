# Data processing imports
import scipy.io as io
import numpy as np
from pyDOE import lhs

# Plotting imports
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec

def load_dataset(file):
    data = io.loadmat(file)
    return data['x'], data['t'], data['usol'].T

# Inference

def preprocess_data_discrete_inference(file, idx_t0, idx_t1, q = 500, N = 250, noise = 0.0):
    x, t, u_exact = load_dataset(file)
    X, T = np.meshgrid(x, t)
    test_X = x
    test_u = u_exact[idx_t1, :]
    
    # Compute domain bounds for x
    lb = test_X.min(0)
    ub = test_X.max(0)
    
    # Determine dt
    dt = t[idx_t1] - t[idx_t0]
    
    # Sampling for initial step
    idx_x = np.random.choice(x.shape[0], N, replace = False)
    x0 = x[idx_x,:]
    u0 = u_exact[idx_t0:idx_t0+1, idx_x].T
    u0 = u0 + noise*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])

    x1 = np.vstack([lb, ub])
    
    tmp = np.float32(np.loadtxt(f'IRK_weights/Butcher_IRK{q}.txt', ndmin = 2))
    IRK_weights = np.reshape(tmp[:q**2+q], (q+1,q))

    return x, t, u_exact, T, lb, ub, dt, x0, u0, x1, test_X, test_u, IRK_weights
def plot_results_discrete_inference(x, t, x0, u0, u_exact, test_X, u1_pred, idx_t0, idx_t1, lb, ub):
    fig = plt.figure(figsize = (10, 9.5))
    ax = plt.gca()
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ####### Row 0: h(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/2 + 0.1, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(u_exact.T, interpolation='nearest', cmap='rainbow', 
                  extent=[t.min(), t.max(), test_X.min(), test_X.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
        
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[idx_t0]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[idx_t1]*np.ones((2,1)), line, 'w-', linewidth = 1)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    
    ax.set_title('$u(t,x)$', fontsize = 10)
    
    
    ####### Row 1: h(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=1-1/2-0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)
    
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,u_exact[idx_t0,:], 'b-', linewidth = 2) 
    ax.plot(x0, u0, 'rx', linewidth = 2, label = 'Data')      
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = %.2f$' % (t[idx_t0]), fontsize = 10)
    ax.set_xlim([lb-0.1, ub+0.1])
    ax.legend(loc='upper center', bbox_to_anchor=(0.8, -0.3), ncol=2, frameon=False)

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,u_exact[idx_t1,:], 'b-', linewidth = 2, label = 'Exact') 
    ax.plot(test_X, u1_pred[:,-1], 'r--', linewidth = 2, label = 'Prediction')      
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = %.2f$' % (t[idx_t1]), fontsize = 10)    
    ax.set_xlim([lb-0.1, ub+0.1])
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.1, -0.3), ncol=2, frameon=False)
    plt.show()


# Identification    
    
def preprocess_data_discrete_identification(file, idx_t0, idx_t1, N0 = 250, N1 = 250, noise = 0.0):
    x, t, u_exact = load_dataset(file)
    
    # Compute domain bounds for x
    lb = x.min(0)
    ub = x.max(0)
    
    # Determine dt
    dt = t[idx_t1] - t[idx_t0]
    
    # Determine q
    q = int(np.ceil(0.5*np.log(np.finfo(float).eps)/np.log(dt)))
    
    # Sampling for initial step
    idx_x = np.random.choice(x.shape[0], N0, replace = False)
    x0 = x[idx_x,:]
    u0 = u_exact[idx_t0:idx_t0+1, idx_x].T
    u0 = u0 + noise*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])
    
    # Sampling for final step
    idx_x = np.random.choice(x.shape[0], N1, replace = False)
    x1 = x[idx_x,:]
    u1 = u_exact[idx_t1:idx_t1+1, idx_x].T
    u1 = u1 + noise*np.std(u1)*np.random.randn(u1.shape[0], u1.shape[1])
    
    tmp = np.float32(np.loadtxt(f'IRK_weights/Butcher_IRK{q}.txt', ndmin = 2))
    IRK_weights = np.reshape(tmp[:q**2+q], (q+1,q))
    IRK_alphas = IRK_weights[:-1,:]
    IRK_betas = IRK_weights[-1:,:]

    return x, t, u_exact, lb, ub, dt, q, x0, u0, x1, u1, IRK_alphas, IRK_betas

    
def plot_results_discrete_identification(x, t, x0, x1, u_exact, u0, u1, idx_t0, idx_t1, lb, ub, lambda_1, lambda_2):
    fig = plt.figure(figsize = (10, 9.5))
    ax = plt.gca()
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3+0.05, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
        
    h = ax.imshow(u_exact.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(),t.max(), lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[idx_t0]*np.ones((2,1)), line, 'w-', linewidth = 1.0)
    ax.plot(t[idx_t1]*np.ones((2,1)), line, 'w-', linewidth = 1.0)    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('$u(t,x)$', fontsize = 10)
    
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=1-1/3-0.1, bottom=1-2/3, left=0.15, right=0.85, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, u_exact[idx_t0,:][:,None], 'b', linewidth = 2, label = 'Exact')
    ax.plot(x0, u0, 'rx', linewidth = 2, label = 'Data')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = %.2f$\n%d trainng data' % (t[idx_t0], u0.shape[0]), fontsize = 10)
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, u_exact[idx_t1,:][:,None], 'b', linewidth = 2, label = 'Exact')
    ax.plot(x1, u1, 'rx', linewidth = 2, label = 'Data')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = %.2f$\n%d trainng data' % (t[idx_t1], u1.shape[0]), fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(-0.3, -0.3), ncol=2, frameon=False)
    
    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=1-2/3-0.05, bottom=0, left=0.15, right=0.85, wspace=0.0)
    
    ax = plt.subplot(gs2[0, 0])
    ax.axis('off')
    ax.text(0.5,0.5,f'Correct PDE: $u_t + u u_x - 0.0031831 u_{{xx}} = 0$ \n$\lambda_1$: {lambda_1:.5f} \t\t $\lambda_2$: {lambda_2:.5f}')
    plt.show() 