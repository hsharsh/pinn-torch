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

def preprocess_data_continuous_inference(file, Nu = 100, Nf = 10000):
    x, t, u_exact = load_dataset(file)
    X, T = np.meshgrid(x, t)
    test_X = np.hstack([X.flatten()[:,None], T.flatten()[:,None]])
    test_u = u_exact.flatten()[:,None]
    
    # Sampling for initial and boundary conditions
    x_i = np.hstack([X[:1,:].T,T[:1,:].T]) # Initial
    u_i = u_exact[:1,:].T
    x_b1 = np.hstack([X[:,:1], T[:,:1]]) # Boundary 1
    u_b1 = u_exact[:,:1]
    x_b2 = np.hstack([X[:,-1:], T[:,-1:]]) # Boundary 2
    u_b2 = u_exact[:,-1:]
    
    train_X_u = np.vstack([x_i, x_b1, x_b2])
    train_u = np.vstack([u_i, u_b1, u_b2])
    
    # Domain bounds for Lattice Hypercube Sampling
    lb = test_X.min(0)
    ub = test_X.max(0)
    
    collocation_points = lb + (ub-lb)*lhs(2, Nf) # Samples (Nf x 2) points and scales them to be in the domain
    train_X_f = np.vstack([collocation_points, train_X_u])
    
    # Restrics the boundary conditions to only Nu random points 
    sample = np.random.choice(train_X_u.shape[0], size = Nu)
    train_X_u = train_X_u[sample]
    train_u = train_u[sample]
    
    return x, t, u_exact, X, T, lb, ub, train_X_u, train_u, train_X_f, test_X, test_u

def plot_results_continuous_inference(x, t, X, T, u_exact, u_pred, train_X_u, train_X_f, train_u, test_X):
    
    u_pred = griddata(test_X, u_pred.flatten(), (X, T), method='cubic')

    fig = plt.figure(figsize = (10, 9.5))
    ax = plt.gca()
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ####### Row 0: u(t,x) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])


    h = ax.imshow(u_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(train_X_u[:,1], train_X_u[:,0], 'kx', markersize = 4, clip_on = False)
    ax.plot(train_X_f[:,1], train_X_f[:,0], 'k.', markersize = 1, clip_on = False)

    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
#     ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u(t,x)$', fontsize = 10)


    ####### Row 1: u(t,x) slices ##################
    gs1 = gridspec.GridSpec(2, 3)
    gs1.update(top=1-12/30, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,u_exact[0,:], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x,u_pred[0,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = 0$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,u_exact[24,:], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x,u_pred[24,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = 0.25$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,u_exact[49,:], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x,u_pred[49,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 0.50$', fontsize = 10)

    ax = plt.subplot(gs1[1, 0])
    ax.plot(x,u_exact[74,:], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x,u_pred[74,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 0.75$', fontsize = 10)

    ax = plt.subplot(gs1[1, 1])
    ax.plot(x,u_exact[99,:], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x,u_pred[99,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 1.00$', fontsize = 10)

    ax.legend(loc='center', bbox_to_anchor=(2.5, 0.6), ncol=5, frameon=False)
    plt.show()


# Identification

    
def preprocess_data_continuous_identification(file, N = 2000, noise = 0.0):
    x, t, u_exact = load_dataset(file)
    X, T = np.meshgrid(x, t)
    test_X = np.hstack([X.flatten()[:,None], T.flatten()[:,None]])
    test_u = u_exact.flatten()[:,None]
        
    # Domain bounds for Lattice Hypercube Sampling
    lb = test_X.min(0)
    ub = test_X.max(0)
    
    # Sample N random points 
    sample = np.random.choice(test_X.shape[0], size = N)
    train_X = test_X[sample]
    train_u = test_u[sample]
    train_u = train_u + noise*np.std(train_u)*np.random.randn(train_u.shape[0], train_u.shape[1])
    
    return x, t, u_exact, X, T, lb, ub, train_X, train_u, test_X, test_u


def plot_results_continuous_identification(x, t, X, T, u_exact, u_pred, train_X, train_u, test_X, lambda_1, lambda_2):
    
    u_pred = griddata(test_X, u_pred.flatten(), (X, T), method='cubic')

    fig = plt.figure(figsize = (10, 9.5))
    ax = plt.gca()
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ####### Row 0: u(t,x) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])


    h = ax.imshow(u_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(train_X[:,1], train_X[:,0], 'k.', markersize = 2, clip_on = False)

    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
#     ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u(t,x)$', fontsize = 10)


    ####### Row 1: u(t,x) slices ##################
    gs1 = gridspec.GridSpec(2, 3)
    gs1.update(top=1-12/30, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,u_exact[0,:], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x,u_pred[0,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = 0$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,u_exact[24,:], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x,u_pred[24,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = 0.25$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,u_exact[49,:], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x,u_pred[49,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 0.50$', fontsize = 10)

    ax = plt.subplot(gs1[1, 0])
    ax.plot(x,u_exact[74,:], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x,u_pred[74,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 0.75$', fontsize = 10)

    ax = plt.subplot(gs1[1, 1])
    ax.plot(x,u_exact[99,:], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x,u_pred[99,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 1.00$', fontsize = 10)
    ax.legend(loc='center', bbox_to_anchor=(1.8, 0.3), ncol=5, frameon=False)   
    
    # Prediction
    ax = plt.subplot(gs1[1, 2])
    ax.axis('off')
    s1 = '$\begin{tabular}{ |c|c| }  \hline Correct PDE & $u_t + u u_x - 0.0031831 u_{xx} = 0$ \\  \hline Identified PDE (clean data) & '
    s2 = '$u_t + %.5f u u_x - %.7f u_{xx} = 0$ \\  \hline ' % (lambda_1, lambda_2)
#     s3 = r'Identified PDE (1\% noise) & '
#     s4 = r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$  \\  \hline ' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s3 = '\end{tabular}$'
    s = s1+s2+s3
    ax.text(-0.3,0.5,f'Correct PDE: $u_t + u u_x - 0.0031831 u_{{xx}} = 0$ \n\t\t\t$\lambda_1$: {lambda_1:.5f}, $\lambda_2$: {lambda_2:.5f}')
        
    plt.show() 
