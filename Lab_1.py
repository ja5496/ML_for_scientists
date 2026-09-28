# Lab 1: MLPs from scratch - Jake Abraham

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

print("jax", jax.__version__, "| devices:", jax.devices())


###########################################################
#                          DATA                           #
###########################################################


FREQS = jnp.array([1.0, 2.0, 4.0, 8.0])
NOISE_STD = 0.02


def target_fn(x):
    """x: (n,) -> (n,)"""
    return jnp.mean(jnp.sin(2 * jnp.pi * FREQS * x[:, None]), axis=-1)


def make_data(key, n=2048, noise_std=NOISE_STD):
    """Returns X: (n, 1), Y: (n, 1)"""
    key_x, key_noise = jax.random.split(key)
    x = jax.random.uniform(key_x, (n,), minval=0.0, maxval=1.0)
    y = target_fn(x) + noise_std * jax.random.normal(key_noise, (n,))
    return x[:, None], y[:, None]


X, Y = make_data(jax.random.key(0))

print(f"target variance : {float(Y.var()):.4f}")   # silly baseline: the error you get by predicting the mean
     

grid = jnp.linspace(0, 1, 1000)
'''
plt.figure(figsize=(7, 2.6))
plt.scatter(np.array(X[:400, 0]), np.array(Y[:400, 0]), s=4, color="#b8b8b4")
plt.plot(np.array(grid), np.array(target_fn(grid)), color="#0b0b0b", lw=1.6)
plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout(); plt.show()
'''


###########################################################
#                          NETWORK                        #
###########################################################

''' 
Given layer sizes, initialize the network with random weights and biases, 
the loss function, a forward pass, and the activation function. 
'''

layer_sizes = [1, 128, 128, 1]

def init_mlp(key, layer_sizes):
    num_layers = len(layer_sizes)-1                             # Number of layers
    keys = jax.random.split(key, num_layers)                # Splitting random key to be different for each layer                            
    params = []
    for k, n_in, n_out in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
        scale = jnp.sqrt(1 / n_in)
        W_init = scale * jax.random.normal(k, shape=(n_in, n_out))
        b_init = jnp.zeros((n_out,))
        params.append((W_init, b_init))  # Random (W,b) for each layer
    return params

def forward(params, x):
    for W, b in params[:-1]:        # Hidden layers
        x = activation(x @ W + b) 
    W, b = params[-1]               # Output layer 
    return x @ W + b                # Prediction     

def activation(x):
    return jnp.tanh(x)

def loss_fn(params, x, y):
    diff = forward(params, x) - y
    mse = jnp.mean(diff**2)
    return mse

###########################################################
#                       OPTIMIZERS                        #
###########################################################

'''
Every optimizer is a pair: init(params) -> state, and
update(params, state, x, y, lr) -> (params, state).
The state carries anything remembered between steps (None if nothing).
'''

'''         

GRADIENT DESCENT:

Regular gradient descent calculates the gradient over the whole training dataset,
and subtracts (learning rate)*(gradient) from the parameters. This is done over many
epochs to optimize weights.


'''
def gd_init(params):
    return None                      # No state

@jax.jit  # Compiles the whole step
def gd_update(params, state, x, y, lr):
    grads = jax.grad(loss_fn)(params, x, y)
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads), state


'''   

STOCHASTIC GRADIENT DESCENT:

Stochastic gradient descent calculates the gradient over random subsets of the 
training dataset, and subtracts (learning rate) * (gradient) from the parameters. 
This allows for many more gradient updates for the same amount of forward passes,
meaning faster convergence. 


'''

sgd_init = gd_init

@jax.jit  # Compiles the whole step
def sgd_update(params, state, x, y, lr): # SAME as gd_update, only difference is batch size
    grads = jax.grad(loss_fn)(params, x, y)
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads), state


'''         

SGD WITH MOMENTUM:        

Calculates the gradient over random subsets of the training dataset. Instead of 
simply subtracting a multiple of the gradient, we subtract a learning rate times 
the "velocity", which is the gradient plus an accumulating momentum vector. "Beta" is
a number between 0 and 1 that decays the velocity/momentum in the presence of no gradient 
so it comes to a stop.


'''
BETA = 0.9                           # Fraction of the previous velocity kept each step

def momentum_init(params):
    return jax.tree_util.tree_map(jnp.zeros_like, params)   # Velocity starts at zero, same shape as params

@jax.jit  # Compiles the whole step
def momentum_update(params, velocity, x, y, lr):
    grads = jax.grad(loss_fn)(params, x, y)
    velocity = jax.tree_util.tree_map(lambda v, g: BETA * v + g, velocity, grads)   # v <- beta*v + g
    params = jax.tree_util.tree_map(lambda p, v: p - lr * v, params, velocity)      # p <- p - lr*v
    return params, velocity


'''              

ADAGRAD:

Stands for 'adaptive gradient descent'. It scales the learning rate for each parameter 
based on past gradients instead of having one constant learning rate. It allows the network 
to learn from more rare signals that would not be picked up by SGD. Uses batch training like 
SGD for convergence speed. Learning rates are divided by the square root of the sum of squares 
of the previous gradients (s). This method has the disadvantage of diminishing learning rates 
because the sum in the denom (s) always increases. 


'''
EPS = 1e-8      # Avoids dividing by zero 

def adagrad_init(params):
    return jax.tree_util.tree_map(jnp.zeros_like, params)   # Running sum of squared gradients, same shape as params

@jax.jit  # Compiles the whole step
def adagrad_update(params, sum_sq, x, y, lr):
    grads = jax.grad(loss_fn)(params, x, y)
    sum_sq = jax.tree_util.tree_map(lambda s, g: s + g**2, sum_sq, grads)                        # G <- G + g^2
    params = jax.tree_util.tree_map(lambda p, g, s: p - lr * g / (jnp.sqrt(s) + EPS), params, grads, sum_sq)  # p <- p - lr*g/sqrt(G)
    return params, sum_sq


'''               

ADAM:               

Adam stands for 'adaptive moment estimation'. It combines momentum with adaptive learning 
rates that use an exponential moving average as opposed to a strict sum. This way, 
it is able to capture rare features and find a deeper minimum without risking learning rate 
collapse. First moment is the moving avg of gradients (like velocity) and second moment is the 
avg of squared gradients (like s). 


'''
BETA1 = 0.9     # Decay rate of the running mean of gradients (momentum)
BETA2 = 0.999   # Decay rate of the running mean of squared gradients (per-parameter scale)

def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return (zeros, zeros, jnp.array(0))                     # (m, v, step count t)

@jax.jit  # Compiles the whole step
def adam_update(params, state, x, y, lr):
    m, v, t = state
    t = t + 1
    grads = jax.grad(loss_fn)(params, x, y)
    m = jax.tree_util.tree_map(lambda m, g: BETA1 * m + (1 - BETA1) * g, m, grads)       # m <- b1*m + (1-b1)*g
    v = jax.tree_util.tree_map(lambda v, g: BETA2 * v + (1 - BETA2) * g**2, v, grads)    # v <- b2*v + (1-b2)*g^2
    m_hat_scale = 1 / (1 - BETA1**t)                        # Bias correction: m and v start at 0, so early
    v_hat_scale = 1 / (1 - BETA2**t)                        # averages are too small without it
    params = jax.tree_util.tree_map(
        lambda p, m, v: p - lr * (m * m_hat_scale) / (jnp.sqrt(v * v_hat_scale) + EPS), params, m, v)  # p <- p - lr*m_hat/sqrt(v_hat)
    return params, (m, v, t)


'''               

MUON:               

Muon is an optimizer only for hidden layers that have square weight matrices. It treats
weight matrices as 2D linear operators, orthogonalizes the update matrix with approximate 
methods, and then scales each of its eigenvalues to 1 via a linear transformation so that 
updates will act on each principal direction (just its eigenvectors) equally. This allows 
for better stability at high learning rates, faster convergence, and runs fast because it 
uses basic matrix multiplication. 

The method below uses Adam for input/output layers because their matrices are not square, 
so it is a hybrid model. 


'''
MUON_MOMENTUM = 0.95
MUON_ADAM_LR = 0.003            # Learning rate for the parameters handled by Adam

def is_hidden_matrix(p):
    return p.ndim == 2 and min(p.shape) > 1         # Only the 128x128 layer here

def newton_schulz(G, steps=5):
    """Approximately orthogonalize G (i.e. U V^T from its SVD) using only matmuls."""
    a, b, c = 3.4445, -4.7750, 2.0315               # Tuned quintic coefficients from the Muon paper
    X = G / (jnp.linalg.norm(G) + 1e-7)             # Scale so all singular values are <= 1
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X         # Pushes every singular value toward 1
    return X.T if transposed else X

def muon_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return (zeros, zeros, zeros, jnp.array(0))      # (Muon momentum, Adam m, Adam v, step count t)

@jax.jit  # Compiles the whole step
def muon_update(params, state, x, y, lr):
    buf, m, v, t = state
    t = t + 1
    grads = jax.grad(loss_fn)(params, x, y)

    # Work leaf by leaf, since each parameter is handled by either Muon or Adam
    p_leaves, treedef = jax.tree_util.tree_flatten(params)
    g_leaves, buf_leaves = treedef.flatten_up_to(grads), treedef.flatten_up_to(buf)
    m_leaves, v_leaves = treedef.flatten_up_to(m), treedef.flatten_up_to(v)

    new_p, new_buf, new_m, new_v = [], [], [], []
    for p, g, bf, mm, vv in zip(p_leaves, g_leaves, buf_leaves, m_leaves, v_leaves):
        if is_hidden_matrix(p):
            bf = MUON_MOMENTUM * bf + g                             # Momentum buffer
            direction = newton_schulz(g + MUON_MOMENTUM * bf)       # Nesterov momentum, then orthogonalize
            scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5        # Shape correction (1.0 for square matrices)
            p = p - lr * scale * direction
        else:
            mm = BETA1 * mm + (1 - BETA1) * g                       # Same as adam_update
            vv = BETA2 * vv + (1 - BETA2) * g**2
            m_hat = mm / (1 - BETA1**t)
            v_hat = vv / (1 - BETA2**t)
            p = p - MUON_ADAM_LR * m_hat / (jnp.sqrt(v_hat) + EPS)
        new_p.append(p); new_buf.append(bf); new_m.append(mm); new_v.append(vv)

    unflatten = lambda leaves: jax.tree_util.tree_unflatten(treedef, leaves)
    return unflatten(new_p), (unflatten(new_buf), unflatten(new_m), unflatten(new_v), t)

###########################################################
#                      TRAINING LOOP                      #
###########################################################

eval_loss = jax.jit(loss_fn)
n_epochs = 400
lr = 0.01
n = X.shape[0]

def train(init_fn, update_fn, batch_size, n_epochs=n_epochs, lr=lr):
    """Same init and shuffle seed for every run, so only the optimizer/batch size differs."""
    params = init_mlp(jax.random.key(1), layer_sizes)
    opt_state = init_fn(params)
    key = jax.random.key(2)                   # Use a separate key for shuffling
    losses = []

    for epoch in range(n_epochs):
        key, k = jax.random.split(key)
        perm = jax.random.permutation(k, n)   # Random ordering for shuffle
        X_shuf, Y_shuf = X[perm], Y[perm]     # Permute X and Y indices to shuffle for training

        for i in range(0, n, batch_size):
            xb = X_shuf[i : i + batch_size]
            yb = Y_shuf[i : i + batch_size]
            params, opt_state = update_fn(params, opt_state, xb, yb, lr)

        losses.append(eval_loss(params, X, Y))
        if epoch % 50 == 0:
            print(epoch, losses[-1])
    return params, losses

runs = {}
print("GD (batch = full dataset)")
runs["GD"] = train(gd_init, gd_update, batch_size=n)
print("SGD (batch = 128)")
runs["SGD"] = train(sgd_init, sgd_update, batch_size=128)
print(f"SGD + momentum (batch = 128, beta = {BETA})")
runs["Momentum"] = train(momentum_init, momentum_update, batch_size=128)
print("Adagrad (batch = 128, lr = 0.1)")
runs["Adagrad"] = train(adagrad_init, adagrad_update, batch_size=128, lr=0.1)   # Adagrad's steps shrink over time, so it needs a larger lr
print("Adam (batch = 128, lr = 0.003)")
runs["Adam"] = train(adam_init, adam_update, batch_size=128, lr=0.003)
print("Muon (batch = 128, lr = 0.02 for the hidden matrix, Adam elsewhere)")
runs["Muon"] = train(muon_init, muon_update, batch_size=128, lr=0.02)


###########################################################
#                          PLOTS                          #
###########################################################

colors = {"GD": "#2a6fdb", "SGD": "#d1495b", "Momentum": "#2a9d5c", "Adagrad": "#e09f1f", "Adam": "#7b4fc9", "Muon": "#1a9fa8"}
fig, (ax_loss, ax_fit) = plt.subplots(1, 2, figsize=(15, 6))

# Loss curves vs. baseline and noise floor
for name, (params, losses) in runs.items():
    ax_loss.semilogy(np.array(losses), color=colors[name], lw=1.4, label=name)
ax_loss.axhline(float(Y.var()), color="#b8b8b4", ls="--", label="predict mean")
ax_loss.axhline(NOISE_STD**2, color="#0b0b0b", ls=":", label="noise floor")
ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("MSE"); ax_loss.legend()

# Learned functions vs. target
ax_fit.scatter(np.array(X[:400, 0]), np.array(Y[:400, 0]), s=4, color="#b8b8b4")
ax_fit.plot(np.array(grid), np.array(target_fn(grid)), color="#0b0b0b", lw=1.6, label="target")
for name, (params, losses) in runs.items():
    ax_fit.plot(np.array(grid), np.array(forward(params, grid[:, None])[:, 0]), color=colors[name], lw=1.6, label=name)
ax_fit.set_xlabel("x"); ax_fit.set_ylabel("y"); ax_fit.legend()

plt.tight_layout(); plt.show()
