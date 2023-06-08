# %%
import numpy as np

# %% Helper functions

def lorenz(X, bif_arg = None):
    
    if bif_arg is None:
        sigma = 10
        beta = 8/3
        rho = 28
    else:
        sigma = bif_arg[0]
        beta = bif_arg[1]
        rho = bif_arg[2]

    x, y, z = X

    dx = sigma*(y-x)
    dy = x*(rho-z)-y
    dz = x*y-beta*z

    output = np.array([dx, dy, dz])
    return output

def lorenz_modified(X):
    sigma = 10
    beta = 8/3
    rho = 32

    x, y, z = X

    dx = sigma*(y-x)
    dy = x*(rho-z)-y
    dz = x*y-beta*z

    output = np.array([dx, dy, dz])
    return output

def periodic_lorenz(X):
    sigma = 10
    beta = 8/3
    rho = 160

    x, y, z = X

    dx = sigma*(y-x)
    dy = x*(rho-z)-y
    dz = x*y-beta*z

    output = np.array([dx, dy, dz])
    return output

def lorenz_dyn_noise(X, bif_arg = None):
    
    if bif_arg is None:
        sigma = 10
        beta = 8/3
        rho = 28
    else:
        sigma = bif_arg[0]
        beta = bif_arg[1]
        rho = bif_arg[2]

    x, y, z = X

    dyn_noise = 1+0.05*(np.random.rand()-0.5)
    dx = sigma*(y-x)
    dy = x*(rho-z)-y
    dz = x*y-beta*z

    output = np.array([dx, dy, dz])*dyn_noise
    return output
# %%
def rossler(X):
    a = 0.2
    b = 0.2
    c = 5.7

    x, y, z = X

    dx = -y-z
    dy = x+a*y
    dz = b + z*(x-c)

    output = np.array([dx, dy, dz])
    return output

def rossler_modified(X):
    a = 0.3
    b = 0.2
    c = 5.7

    x, y, z = X

    dx = -y-z
    dy = x+a*y
    dz = b + z*(x-c)

    output = np.array([dx, dy, dz])
    return output

def rossler_banded(X):
    alpha = 0.5
    a = 0.2 + 0.09*alpha
    b = 0.2 - 0.06*alpha
    c = 5.7 - 1.18*alpha

    x, y, z = X

    dx = -y-z
    dy = x+a*y
    dz = b + z*(x-c)

    output = np.array([dx, dy, dz])
    return output

def rossler_nonbanded(X):
    alpha = -0.25
    a = 0.2 + 0.09*alpha
    b = 0.2 - 0.06*alpha
    c = 5.7 - 1.18*alpha

    x, y, z = X

    dx = -y-z
    dy = x+a*y
    dz = b + z*(x-c)

    output = np.array([dx, dy, dz])
    return output

# Phase coherent Rossler
def rossler_PC(X, bif_arg = None):
    a = 0.165
    b = 0.4
    c = 8.5

    x, y, z = X

    dx = -y-z
    dy = x+a*y
    dz = b + z*(x-c)

    output = np.array([dx, dy, dz])
    return output

# Non-phase coherent rossler
def rossler_NPC(X, bif_arg = None):
    a = 0.265
    b = 0.4
    c = 8.5

    x, y, z = X

    dx = -y-z
    dy = x+a*y
    dz = b + z*(x-c)

    output = np.array([dx, dy, dz])
    return output

def chua(X, bif_arg = None):

    #Parameters
    if bif_arg is None:
        k = -1
        beta = 53.612186
        gamma = -0.75087096
        alpha = 17
    else:
        k = -1
        beta = 53.612186
        gamma = -0.75087096
        alpha = bif_arg[0]
    
    

    #Cubic nonlinearity
    a = -0.0375582129
    b = -0.8415410391
    def f(x):
        output = a*x**3 + b*x
        return output

    x,y,z = X
    dx = k*(y-x+z)
    dy = k*alpha*(x-y-f(y))
    dz = k*(-beta*x-gamma*z)

    output = np.array([dx, dy, dz])
    return output

def halvorsen(X):
    #Parameters Xiaogen Yin, YJ Cao, Sychronisation of Chua’s oscillator via the state observer techniqu
    a = 1.27
    b = 4

    x,y,z = X
    dx = -a*x-b*(y+z)-y**2
    dy = -a*y-b*(z+x)-z**2
    dz = -a*z-b*(x+y)-x**2

    output = np.array([dx, dy, dz])
    return output

def duffing(X):
    #Chaotic Duffing oscillator (Guckenheimer, Kanamaru)
    alpha = 1
    beta = -1
    delta = 0.2
    gamma = 0.3
    omega = 1

    x,xdot,psi = X
    dx = xdot
    dy = -delta*xdot-beta*x-alpha*x**3+gamma*np.cos(psi)
    dz = omega

    output = np.array([dx, dy, dz])
    return output

def pendulum(X):
    #Forced pendulum intermittency (Grebogi 1987)
    ohm = 1
    omega = 1
    v = 0.22
    p = 2.8

    phi, phidot, psi = X
    dx = phidot
    dy = -v*phidot-(ohm^2)*np.sin(phi)+p*np.cos(psi)
    dz = omega

    output = np.array([dx, dy, dz])
    return output

def FHN_neuron2(X, f = 0.12643):
    # f = 0.12643
    alpha = 0.1 #input Driving amplitude
    b1 = 10
    b2 = 1

    v, w, theta = X
    omega = 2*np.pi*f
    dv = v*(v-1)*(1-b1*v) - w + (alpha/omega)*np.cos(theta) #Sodium positive depolarisation channel
    dw = b2*v
    dtheta = omega
    # dI = dI
    # ddI = -(ω^2)*I

    # output = [dv, dw, dI, ddI]
    output = [dv, dw, dtheta]
    return output

def yu(X):
    # Yu's double attractor
    a = 2.5

    x,y,z = X

    dx = -x+0.5*x*z+y*z
    dy = a*y-1.2*x*z
    dz = x*y-6*z

    output = np.array([dx, dy, dz])
    return output

def huang(X):
    # 4D Chaos (Huang et al., 2019)

    a = 6
    b = 11
    c = 5

    x,y,z,w = X

    dx = a*(y-x)
    dy = x*z+w
    dz = b-x*y
    dw = y*z-c*w

    output = np.array([dx, dy, dz, dw])
    return output


# %% RK4 Integration

def linearise(X, model = None, bif_arg = None):

    if model is None:
        output = lorenz(X, bif_arg = bif_arg)
    else:
        output = model(X, bif_arg = bif_arg)

    return output


def integrate(model, bif_arg = None, dt=0.02, T=5000, RK = False, supersample = 1, dims = 3, init = None, noise = None):
    # Generate Empty Matrix
    S = np.zeros((T*supersample, dims))
    dS = np.zeros(dims)
    del_t = (dt/supersample)

    # Initialise starting values
    if init is None:
        S[0, :] = 1*(np.random.rand(np.shape(S[0, :])[0]) - 0.5)
    else:
        S[0, :] = init

    for t in range(T*supersample):
        if not RK:
            dS[:] = linearise(S[t, :], model = model, bif_arg = bif_arg)
        else:
            # Runge Kutta Integration
            k1 = linearise(S[t, :], model = model, bif_arg = bif_arg)
            k2 = linearise(S[t, :] +del_t*(k1/2), model = model, bif_arg = bif_arg)
            k3 = linearise(S[t, :] +del_t*(k2/2), model = model, bif_arg = bif_arg)
            k4 = linearise(S[t, :] +del_t*k3, model = model, bif_arg = bif_arg)
            dS[:] = (k1+2*(k2+k3)+k4)/6

        if noise is not None:
            dS = dS + noise*np.random.randn(dims)

        if t < T*supersample-1:
            S[t+1, :] = S[t, :]+del_t*dS
        
    return S[np.arange(0,(T*supersample),supersample, dtype = int),:]

# function MackeyGlass(T, dt, lag)
#     γ = 1
#     β = 2
#     n = 9.65
#     τ = Int(lag/dt)
#     x0 = 0.5
#     x = Array{Float64}(undef, T+wash)
#     x[1:(τ+1)] .= x0
#     for i in (τ+1):(length(x)-1)
#         dx = β*((x[i-τ])/(1+x[i-τ]^n))-γ*x[i]
#         x[i+1] = x[i] + dt*dx
#     end
#     return x
# end

# # %% Time Delay Embedding
# function custom_embed(x, τ, m)
#     S = zeros(length(x)-(m-1)*τ, m)

#     for dim in 1:m
#         S[:,dim] = x[1+(m-dim)*τ:end-(dim-1)*τ]
#     end

#     return S
# end

def nonunif_embed(x, lags):
    # Sort lags in ascending order
    tau_array = np.sort(lags)

    # Create storage array for result
    m = len(tau_array)+1
    # S = zeros(length(x)-τ_array[end], m)
    S = np.zeros((len(x)-np.sum(tau_array), m))

    # Non-uniform embedding
    for dim in range(m):
        if dim == 0:
            S[:,dim] = x[np.sum(tau_array):]
        elif dim < m:
            S[:,dim] = x[(np.sum(tau_array)-np.sum(tau_array[0:(dim)])):(-np.sum(tau_array[:(dim)]))]
        else:
            S[:,dim] = x[0:(-np.sum(tau_array))]

    return S

# Same as nonunif_embed except tau lags are calculated relative to the first component, and not successive
def nonunif_embed2(x, lags):
    # Sort lags in ascending order
    tau_array = np.sort(lags)

    # Create storage array for result
    m = len(tau_array)+1
    S = np.zeros((len(x)-tau_array[-1], m))

    # Non-uniform embedding
    for dim in range(m):
        if dim == 0:
            S[:,dim] = x[tau_array[-1]:]
        else:
            S[:,dim] = x[tau_array[-1]-tau_array[dim-1]:-tau_array[dim-1]]

    return S

# %%
