# Lab 2: Transformers from scratch - Jake Abraham

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

print("jax", jax.__version__, "| devices:", jax.devices())


###########################################################
#                          DATA                           #
###########################################################

'''
Every training example is a different function, drawn from a Gaussian process with a
random lengthscale. The network only sees a handful of noisy points from each one, so
nothing about any one function can be stored in the weights: it has to work it out in
context. Since the functions come from a GP, the best possible answer is known exactly
(gp_posterior), which gives us something to compare against.
'''

X_MAX = 2.0
NOISE_STD = 0.05
ELL_MIN, ELL_MAX = 0.2, 1.0


def rbf(x1, x2, ell):
    """x1: (b, n), x2: (b, m), ell: (b,) -> (b, n, m)"""
    return jnp.exp(-0.5 * (x1[:, :, None] - x2[:, None, :]) ** 2 / ell[:, None, None] ** 2)


def sample_tasks(key, batch, n_ctx, n_tgt=32, ell=None, x_tgt=None):
    """Draw `batch` random functions, each observed (with noise) at n_ctx context and n_tgt target points.
    Returns xc, yc: (batch, n_ctx)   xt, yt: (batch, n_tgt)   ell: (batch,)"""
    k_ell, k_xc, k_xt, k_y = jax.random.split(key, 4)
    if ell is None:
        ell = jnp.exp(jax.random.uniform(k_ell, (batch,), minval=np.log(ELL_MIN), maxval=np.log(ELL_MAX)))
    ell = jnp.broadcast_to(jnp.asarray(ell, jnp.float32), (batch,))
    xc = jax.random.uniform(k_xc, (batch, n_ctx), minval=-X_MAX, maxval=X_MAX)
    if x_tgt is None:
        xt = jax.random.uniform(k_xt, (batch, n_tgt), minval=-X_MAX, maxval=X_MAX)
    else:
        xt = jnp.broadcast_to(x_tgt, (batch, x_tgt.shape[-1]))
    x = jnp.concatenate([xc, xt], axis=1)
    K = rbf(x, x, ell) + (NOISE_STD**2 + 1e-5) * jnp.eye(x.shape[1])
    y = jnp.einsum("bij,bj->bi", jnp.linalg.cholesky(K), jax.random.normal(k_y, x.shape))
    return xc, y[:, :n_ctx], xt, y[:, n_ctx:], ell


def gp_posterior(xc, yc, xt, ell):
    """Exact GP prediction at xt given (xc, yc), if you knew the lengthscale.
    Returns mean: (b, n_tgt), cov: (b, n_tgt, n_tgt)"""
    ell = jnp.broadcast_to(jnp.asarray(ell, jnp.float32), (xc.shape[0],))
    Kcc = rbf(xc, xc, ell) + NOISE_STD**2 * jnp.eye(xc.shape[1])
    Ktc = rbf(xt, xc, ell)
    Ktt = rbf(xt, xt, ell) + NOISE_STD**2 * jnp.eye(xt.shape[1])
    A = jnp.linalg.solve(Kcc, jnp.swapaxes(Ktc, 1, 2))
    mean = jnp.einsum("bct,bc->bt", A, yc)
    return mean, Ktt - Ktc @ A


xc, yc, xt, yt, ell = sample_tasks(jax.random.key(0), batch=1000, n_ctx=10)
mean, cov = gp_posterior(xc, yc, xt, ell)

print(f"target variance               : {float(yt.var()):.4f}")   # Silly baseline: predict 0 everywhere
print(f"GP error, 10 points, true ell : {float(((mean - yt) ** 2).mean()):.4f}")   # The best you can do


grid = jnp.linspace(-X_MAX, X_MAX, 400)
'''
fig, axes = plt.subplots(1, 3, figsize=(10, 2.6), sharey=True)
for ax, ell_, key in zip(axes, [0.2, 0.45, 1.0], jax.random.split(jax.random.key(1), 3)):
    xc, yc, xt, yt, _ = sample_tasks(key, 1, n_ctx=10, ell=ell_, x_tgt=grid)
    mean, cov = gp_posterior(xc, yc, grid[None], ell_)
    std = jnp.sqrt(jnp.diagonal(cov[0]))
    ax.fill_between(np.array(grid), np.array(mean[0] - 2 * std), np.array(mean[0] + 2 * std), color="#e4e4e0")
    ax.plot(np.array(grid), np.array(yt[0]), color="#b8b8b4", lw=1)
    ax.plot(np.array(grid), np.array(mean[0]), color="#0b0b0b", lw=1.6)
    ax.scatter(np.array(xc[0]), np.array(yc[0]), s=14, color="#0b0b0b", zorder=3)
    ax.set_title(f"lengthscale {ell_}", fontsize=10); ax.set_xlabel("x")
axes[0].set_ylabel("y"); plt.tight_layout(); plt.show()
'''


###########################################################
#                        ATTENTION                        #
###########################################################

'''

TOKENS:

Every observed point becomes a token (x, y, 1), and every point where we want a prediction
becomes a token (x, 0, 0). The last entry is a flag, so the network can tell a target apart
from a context point that just happens to have y = 0.


'''

def make_tokens(xc, yc, xt):
    """xc, yc: (b, n_ctx), xt: (b, n_tgt) -> (b, n_ctx + n_tgt, 3)"""
    ctx = jnp.stack([xc, yc, jnp.ones_like(xc)], axis=-1)                       # (x, y, 1)
    tgt = jnp.stack([xt, jnp.zeros_like(xt), jnp.zeros_like(xt)], axis=-1)      # (x, 0, 0)
    return jnp.concatenate([ctx, tgt], axis=1)


'''

THE MASK:

mask[i, j] says whether token i may look at token j. Here every token looks at the context
tokens only:

    - Targets look at the context, which is where all the information about the function is.
    - Targets do NOT look at other targets. A target token carries no y, so there is nothing
      to learn from it, and if targets talked to each other the prediction at one x would
      depend on which other x's we happened to ask about. With this mask, each target's
      prediction depends only on its own x and the context, so we can query 1 point or
      1000 points and get the same answer at each one.
    - Context tokens don't look at targets either, for the same reason.

A target still knows its own x, since it is carried along by the residual connection.


'''

def make_mask(n_ctx, n_tgt):
    """-> (n, n) with n = n_ctx + n_tgt, True where token i (row) may look at token j (column)"""
    is_ctx = jnp.arange(n_ctx + n_tgt) < n_ctx
    return jnp.broadcast_to(is_ctx[None, :], (n_ctx + n_tgt, n_ctx + n_tgt))   # Every row: all context, no targets


'''

MULTI-HEAD SELF-ATTENTION:

Every token makes a query (what am I looking for?), a key (what do I have?) and a value
(what do I pass on?). Token i's output is a weighted average of the values of the tokens
it is allowed to look at, with weights softmax(q_i . k_j / sqrt(d_head)), so tokens whose
keys match its query count the most. Splitting the width into several smaller heads lets
each head look for something different (e.g. nearby points vs. the overall size of the
function). The heads are glued back together and mixed by one last matrix, Wo.


'''
N_HEADS = 4

def init_attention(key, d):
    keys = jax.random.split(key, 4)
    scale = jnp.sqrt(1 / d)
    return {name: scale * jax.random.normal(k, (d, d)) for name, k in zip(["Wq", "Wk", "Wv", "Wo"], keys)}

def split_heads(z):
    """(b, n, d) -> (b, n, heads, d_head)"""
    return z.reshape(*z.shape[:-1], N_HEADS, z.shape[-1] // N_HEADS)

def attention_weights(params, h, mask):
    """h: (b, n, d), mask: (n, n) -> (b, heads, n, n), row i = how much token i looks at each token j"""
    q, k = split_heads(h @ params["Wq"]), split_heads(h @ params["Wk"])
    scores = jnp.einsum("bihd,bjhd->bhij", q, k) / jnp.sqrt(q.shape[-1])
    scores = jnp.where(mask, scores, -jnp.inf)          # Forbidden pairs get exactly zero weight after the softmax
    return jax.nn.softmax(scores, axis=-1)              # Each row sums to 1

def attention(params, h, mask):
    """h: (b, n, d), mask: (n, n) -> (b, n, d)"""
    weights = attention_weights(params, h, mask)
    v = split_heads(h @ params["Wv"])
    out = jnp.einsum("bhij,bjhd->bihd", weights, v)     # Weighted average of the values, for each head
    return out.reshape(h.shape) @ params["Wo"]          # Glue heads back together and mix them


'''

TRANSFORMER BLOCK:

Layer norm -> attention -> residual, then layer norm -> MLP -> residual. Attention is the
only place where tokens exchange information; the MLP then processes each token on its own.
The residual connections mean each block only has to learn a correction to what it is given,
and the layer norms keep every token at a sensible scale no matter how deep we go.

The MLP inside the block is the same one as Lab 1, with a GELU (a smooth ReLU) activation.


'''

def init_mlp(key, layer_sizes):
    num_layers = len(layer_sizes)-1                         # Number of layers
    keys = jax.random.split(key, num_layers)                # Splitting random key to be different for each layer
    params = []
    for k, n_in, n_out in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
        scale = jnp.sqrt(1 / n_in)
        W_init = scale * jax.random.normal(k, shape=(n_in, n_out))
        b_init = jnp.zeros((n_out,))
        params.append((W_init, b_init))  # Random (W,b) for each layer
    return params

def mlp(params, x):
    for W, b in params[:-1]:        # Hidden layers
        x = activation(x @ W + b)
    W, b = params[-1]               # Output layer
    return x @ W + b

def activation(x):
    return jax.nn.gelu(x)

def init_layer_norm(d):
    return {"gain": jnp.ones((d,)), "bias": jnp.zeros((d,))}

def layer_norm(params, h):
    mu = h.mean(axis=-1, keepdims=True)
    var = h.var(axis=-1, keepdims=True)
    return params["gain"] * (h - mu) / jnp.sqrt(var + 1e-5) + params["bias"]    # Zero mean, unit variance per token, then rescale

def init_block(key, d):
    k_attn, k_mlp = jax.random.split(key)
    return {"ln1": init_layer_norm(d), "attn": init_attention(k_attn, d),
            "ln2": init_layer_norm(d), "mlp": init_mlp(k_mlp, [d, 4 * d, d])}

def block(params, h, mask):
    h = h + attention(params["attn"], layer_norm(params["ln1"], h), mask)     # Tokens exchange information
    h = h + mlp(params["mlp"], layer_norm(params["ln2"], h))                  # Each token thinks about it on its own
    return h


###########################################################
#                     THE TRANSFORMER                     #
###########################################################

'''
Embed every token with a small MLP, run them through a stack of transformer blocks, and
read out a mean and a log-variance at each target token. Predicting a variance as well
as a mean lets the model say how unsure it is (e.g. far from any context point).

Note there is no positional encoding: the transformer sees the context as a SET of points.
'''
D_MODEL = 64
N_LAYERS = 4

def init_transformer(key, d=D_MODEL, n_layers=N_LAYERS):
    k_embed, k_head, *k_blocks = jax.random.split(key, n_layers + 2)
    return {"embed": init_mlp(k_embed, [3, d, d]),
            "blocks": [init_block(k, d) for k in k_blocks],
            "ln_out": init_layer_norm(d),
            "head": init_mlp(k_head, [d, d, 2])}

def transformer(params, xc, yc, xt):
    """xc, yc: (b, n_ctx), xt: (b, n_tgt) -> mean, log_var: (b, n_tgt)"""
    n_ctx, n_tgt = xc.shape[1], xt.shape[1]
    h = mlp(params["embed"], make_tokens(xc, yc, xt))                        # (b, n_ctx + n_tgt, d)
    mask = make_mask(n_ctx, n_tgt)
    for p in params["blocks"]:
        h = block(p, h, mask)
    out = mlp(params["head"], layer_norm(params["ln_out"], h[:, n_ctx:]))    # Read out at the targets only
    return out[..., 0], out[..., 1]


'''

LOSS:

Gaussian negative log-likelihood of the targets under the predicted mean and variance.
The (y - mean)^2 / var term punishes being wrong, more so when the model claimed to be
sure, and the log(var) term stops it from just claiming a huge variance everywhere.


'''

def gaussian_nll(mean, log_var, y):
    return 0.5 * jnp.mean(log_var + (y - mean) ** 2 / jnp.exp(log_var) + jnp.log(2 * jnp.pi))

def loss_fn(params, model, xc, yc, xt, yt):
    mean, log_var = model(params, xc, yc, xt)
    return gaussian_nll(mean, log_var, yt)


###########################################################
#                 AN MLP, FOR COMPARISON                  #
###########################################################

'''
The MLP gets the same job, but has no idea that the context is a set. The context is
flattened into one long vector, zero-padded up to 30 points, with a 0/1 mask so it can tell
padding apart from a real point at (0, 0), and the query x is appended at the end. Each
target gets its own forward pass.

Two things are built into the transformer that the MLP has to do without:
    - It has to LEARN that the order of the context points doesn't matter.
    - It has a fixed number of input slots, so it can never use more than 30 points.

Sizes are picked so it has about as many parameters as the transformer.
'''
MAX_CTX = 30

def mlp_inputs(xc, yc, xt):
    """xc, yc: (b, n_ctx), xt: (b, n_tgt) -> (b, n_tgt, 3 * MAX_CTX + 1)"""
    b, n_ctx = xc.shape
    n_keep = min(n_ctx, MAX_CTX)                                                 # Anything past MAX_CTX doesn't fit, so it's dropped
    pad = lambda z: jnp.pad(z[:, :n_keep], ((0, 0), (0, MAX_CTX - n_keep)))    # Zero-pad up to MAX_CTX
    is_real = jnp.broadcast_to(jnp.arange(MAX_CTX) < n_keep, (b, MAX_CTX)).astype(jnp.float32)
    ctx = jnp.concatenate([pad(xc), pad(yc), is_real], axis=-1)                 # (b, 3 * MAX_CTX)
    ctx = jnp.broadcast_to(ctx[:, None, :], (b, xt.shape[1], ctx.shape[-1]))   # Same context for every target
    return jnp.concatenate([ctx, xt[..., None]], axis=-1)                       # Append the query x

mlp_sizes = [3 * MAX_CTX + 1, 256, 256, 256, 256, 2]

def mlp_model(params, xc, yc, xt):
    """xc, yc: (b, n_ctx), xt: (b, n_tgt) -> mean, log_var: (b, n_tgt)"""
    out = mlp(params, mlp_inputs(xc, yc, xt))
    return out[..., 0], out[..., 1]


###########################################################
#                        OPTIMIZER                        #
###########################################################

'''

ADAM (from Lab 1), with two additions that help transformers train smoothly:

    - Learning rate schedule: a linear warmup for the first steps (Adam's second moment
      estimate is poor early on, so big early steps can blow up), then a cosine decay
      to zero, so the last steps are small and the model settles into a minimum.
    - Gradient clipping: if the gradient norm is bigger than 1, scale it down to 1.
      The NLL can spike when the model is very sure and very wrong, and this stops one
      bad batch from undoing a lot of training.


'''
BETA1 = 0.9     # Decay rate of the running mean of gradients (momentum)
BETA2 = 0.999   # Decay rate of the running mean of squared gradients (per-parameter scale)
EPS = 1e-8      # Avoids dividing by zero
PEAK_LR = 1e-3
WARMUP = 1000
N_STEPS = 10000                 # ~11 min for the transformer and ~3 min for the MLP on a laptop CPU
CLIP = 1.0

def learning_rate(t):
    warmup = t / WARMUP
    cosine = 0.5 * (1 + jnp.cos(jnp.pi * (t - WARMUP) / (N_STEPS - WARMUP)))
    return PEAK_LR * jnp.where(t < WARMUP, warmup, cosine)

def clip_by_global_norm(grads):
    norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))   # Norm of ALL the gradients together
    return jax.tree_util.tree_map(lambda g: g * jnp.minimum(1.0, CLIP / (norm + EPS)), grads)

def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return (zeros, zeros, jnp.array(0))                     # (m, v, step count t)

def adam_update(params, grads, state):
    m, v, t = state
    t = t + 1
    lr = learning_rate(t)
    m = jax.tree_util.tree_map(lambda m, g: BETA1 * m + (1 - BETA1) * g, m, grads)       # m <- b1*m + (1-b1)*g
    v = jax.tree_util.tree_map(lambda v, g: BETA2 * v + (1 - BETA2) * g**2, v, grads)    # v <- b2*v + (1-b2)*g^2
    m_hat_scale = 1 / (1 - BETA1**t)                        # Bias correction: m and v start at 0, so early
    v_hat_scale = 1 / (1 - BETA2**t)                        # averages are too small without it
    params = jax.tree_util.tree_map(
        lambda p, m, v: p - lr * (m * m_hat_scale) / (jnp.sqrt(v * v_hat_scale) + EPS), params, m, v)  # p <- p - lr*m_hat/sqrt(v_hat)
    return params, (m, v, t)


###########################################################
#                      TRAINING LOOP                      #
###########################################################

'''
Every step draws a brand new batch of functions, with a random number of context points
between 3 and 30, and 32 targets. The model never sees the same function twice, so there
is nothing to memorize and no overfitting: it can only get better at the inference itself.

The number of context points changes the shapes, so jax.jit compiles the step once per
value (28 of them), and then reuses it.
'''
BATCH = 128
N_TGT = 32
CTX_MIN, CTX_MAX = 3, 30

@partial(jax.jit, static_argnums=(0, 4))  # Compiles the whole step, once per model and number of context points
def train_step(model, params, opt_state, key, n_ctx):
    xc, yc, xt, yt, _ = sample_tasks(key, BATCH, n_ctx, N_TGT)                # Fresh functions every step
    loss, grads = jax.value_and_grad(loss_fn)(params, model, xc, yc, xt, yt)
    params, opt_state = adam_update(params, clip_by_global_norm(grads), opt_state)
    return params, opt_state, loss

def n_params(params):
    return sum(p.size for p in jax.tree_util.tree_leaves(params))

def train(model, params, n_steps=N_STEPS):
    """Same stream of tasks for every model, so only the architecture differs."""
    opt_state = adam_init(params)
    key = jax.random.key(2)                                                     # Use a separate key for the tasks
    n_ctxs = np.array(jax.random.randint(jax.random.key(3), (n_steps,), CTX_MIN, CTX_MAX + 1))   # Context size for every step
    losses = []

    for step in range(n_steps):
        key, k = jax.random.split(key)
        params, opt_state, loss = train_step(model, params, opt_state, k, int(n_ctxs[step]))
        losses.append(loss)
        if step % 1000 == 0:
            print(step, float(np.mean(losses[-1000:])))
    return params, np.array(losses)

models = {"transformer": (transformer, init_transformer(jax.random.key(1))),
          "MLP": (mlp_model, init_mlp(jax.random.key(1), mlp_sizes))}

runs = {}
for name, (model, params) in models.items():
    print(f"{name} ({n_params(params):,} parameters)")
    runs[name] = train(model, params)


###########################################################
#                         RESULTS                         #
###########################################################

@partial(jax.jit, static_argnums=0)  # Compiles the forward pass, once per model and shape
def predict(model, params, xc, yc, xt):
    return model(params, xc, yc, xt)

def mse(mean, y):
    return float(((mean - y) ** 2).mean())

N_TEST = 1000
colors = {"transformer": "#2a6fdb", "MLP": "#d1495b", "GP": "#0b0b0b"}


'''

TEST ERROR VS. NUMBER OF CONTEXT POINTS:

Both models are tested on the same 1000 new functions for every number of context points
from 3 to 80, against the exact GP with the true lengthscale. The GP is the best anyone can
do, and it even gets to cheat a little: it is told the lengthscale, which the networks have
to guess from the points. Nobody can get below the noise floor (NOISE_STD^2), since the
targets themselves are noisy.

    - The transformer follows the GP the whole way down, at about 1.1-1.4x its error.
    - The MLP is about as good as the transformer with 3 points, but gets stuck around
      MSE 0.32 after ~10 points, so it never really learns to combine more than a few of
      its inputs. (Its training loss was still slowly going down, so more steps would
      help, but it is far behind.)
    - Past 30 points, which it never saw in training, the transformer keeps improving,
      since attention is just a weighted average over however many points there are.
      The MLP can't see more than 30, so it stays flat.


'''
ctx_sizes = [3, 4, 6, 8, 11, 16, 22, 30, 40, 56, 80]
errors = {"transformer": [], "MLP": [], "GP": []}

print("n_ctx | transformer | MLP    | GP (true ell)")
for n_ctx in ctx_sizes:
    xc, yc, xt, yt, ell = sample_tasks(jax.random.key(100 + n_ctx), N_TEST, n_ctx, N_TGT)   # Same test functions for every model
    for name, (model, _) in models.items():
        errors[name].append(mse(predict(model, runs[name][0], xc, yc, xt)[0], yt))
    errors["GP"].append(mse(gp_posterior(xc, yc, xt, ell)[0], yt))
    print(f"{n_ctx:5d} | {errors['transformer'][-1]:.4f}      | {errors['MLP'][-1]:.4f} | {errors['GP'][-1]:.4f}")

fig, (ax_train, ax_err) = plt.subplots(1, 2, figsize=(14, 5))

# Training curves, smoothed since the number of context points (and so the loss) jumps around every step
for name, (params, losses) in runs.items():
    ax_train.plot(np.convolve(losses, np.ones(200) / 200, mode="valid"), color=colors[name], lw=1.4, label=name)
ax_train.set_xlabel("step"); ax_train.set_ylabel("training NLL (200-step average)"); ax_train.legend()

# Test error vs. number of context points
ax_err.axvspan(CTX_MIN, CTX_MAX, color="#f0f0ec", label="trained on 3 to 30 points")
for name, errs in errors.items():
    label = "GP, true lengthscale (best possible)" if name == "GP" else name
    ax_err.loglog(ctx_sizes, errs, "o-", color=colors[name], lw=1.6, ms=4, label=label)
ax_err.axhline(NOISE_STD**2, color="#0b0b0b", ls=":", label="noise floor")
ax_err.set_xticks([3, 5, 10, 20, 30, 50, 80], labels=["3", "5", "10", "20", "30", "50", "80"])
ax_err.xaxis.set_minor_formatter(plt.NullFormatter())       # Hides the clashing 4x10^0 style labels
ax_err.set_ylim(1.5e-3, 1)
ax_err.set_xlabel("number of context points"); ax_err.set_ylabel("test MSE"); ax_err.legend()

plt.tight_layout(); plt.show()


'''

SHUFFLING THE CONTEXT:

The transformer has no positional encoding, and attention is a weighted sum over the
context, so shuffling the context just reorders the terms of the sum: the predictions don't
change at all (the ~1e-7 differences are float rounding). This symmetry is built into the
architecture, so it is exact, and it never had to be learned.

The MLP sees the context as a list of numbered slots, so a shuffled context is a completely
different input vector. It was trained on randomly ordered points, so in principle it could
learn that the order doesn't matter, but it only learned it roughly: its predictions move by
~0.24 on average, when its typical error is ~0.6. Averaging the MLP over 32 shufflings
(forcing the symmetry by hand) lowers its MSE from 0.379 to 0.322, so the dependence on the
order is pure noise.


'''

def shuffle_context(key, xc, yc):
    """A different random ordering of the context points for every function in the batch"""
    keys = jax.random.split(key, xc.shape[0])
    perm = jax.vmap(lambda k: jax.random.permutation(k, xc.shape[1]))(keys)
    return jnp.take_along_axis(xc, perm, axis=1), jnp.take_along_axis(yc, perm, axis=1)

xc, yc, xt, yt, ell = sample_tasks(jax.random.key(7), N_TEST, 10, N_TGT)
for name, (model, _) in models.items():
    mean = predict(model, runs[name][0], xc, yc, xt)[0]
    mean_shuf = predict(model, runs[name][0], *shuffle_context(jax.random.key(8), xc, yc), xt)[0]
    print(f"{name:12s}: average change in prediction {float(jnp.abs(mean - mean_shuf).mean()):.1e} | MSE {mse(mean, yt):.4f} -> {mse(mean_shuf, yt):.4f} after shuffling")

# Averaging the MLP over many orderings is a cheap way to make it order-independent
means = [predict(mlp_model, runs["MLP"][0], *shuffle_context(k, xc, yc), xt)[0] for k in jax.random.split(jax.random.key(9), 32)]
print(f"MLP, averaged over 32 shufflings: MSE {mse(jnp.mean(jnp.stack(means), axis=0), yt):.4f}")

# One function, every model shown 25 different orderings of the same 10 points
xc1, yc1, _, yt1, ell1 = sample_tasks(jax.random.key(10), 1, 10, x_tgt=grid)
gp_mean = gp_posterior(xc1, yc1, grid[None], ell1)[0]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
for ax, (name, (model, _)) in zip(axes, models.items()):
    for k in jax.random.split(jax.random.key(11), 25):
        mean = predict(model, runs[name][0], *shuffle_context(k, xc1, yc1), grid[None])[0]
        ax.plot(np.array(grid), np.array(mean[0]), color=colors[name], lw=1, alpha=0.35)
    ax.plot(np.array(grid), np.array(yt1[0]), color="#b8b8b4", lw=1, label="true function")
    ax.plot(np.array(grid), np.array(gp_mean[0]), color="#0b0b0b", lw=1.4, ls="--", label="GP, true lengthscale")
    ax.scatter(np.array(xc1[0]), np.array(yc1[0]), s=16, color="#0b0b0b", zorder=3)
    ax.plot([], [], color=colors[name], lw=1, label=f"{name}, 25 shufflings")
    ax.set_xlabel("x"); ax.legend(loc="lower left", fontsize=9)
axes[0].set_ylabel("y")
plt.tight_layout(); plt.show()


###########################################################
#                PLOTS YOU DIDN'T ASK FOR                 #
###########################################################

'''

1. ATTENTION LEARNED A KERNEL:

Nothing told the transformer to look at nearby points, but most of its heads learned to do
exactly that. On the left, each horizontal line is one context point (at its x), colored by
how much each target on the grid looks at it. All the weight sits along the diagonal: every
target looks mostly at the one or two context points closest to it. That is kernel
regression, which is also what the GP is doing.

On the right is the width of that attention vs. the lengthscale of the function. Like a
GP's kernel, it gets wider for smoother functions (where far away points still tell you
something) and narrower when there are more points to choose from. It only adapts across
the lengthscales it was trained on, and flattens out outside that range. The lengthscale is
never given to the model, so it has to be estimated from the context first. My guess is
that's what the few "global" heads are for (the ones with width ~1.2-1.4 in the printout,
which look at the same points no matter where the target is).


'''

def attention_maps(params, xc, yc, xt):
    """Same as transformer(), but keeps every layer's attention weights -> (b, layers, heads, n, n)"""
    n_ctx, n_tgt = xc.shape[1], xt.shape[1]
    h = mlp(params["embed"], make_tokens(xc, yc, xt))
    mask = make_mask(n_ctx, n_tgt)
    maps = []
    for p in params["blocks"]:
        maps.append(attention_weights(p["attn"], layer_norm(p["ln1"], h), mask))
        h = block(p, h, mask)
    return jnp.stack(maps, axis=1)

def attention_width(maps, xc, xt):
    """How far (in x) each target looks: sqrt(sum_j w_ij (xc_j - xt_i)^2), averaged over targets -> (b, layers, heads)"""
    W = maps[..., xc.shape[1]:, :xc.shape[1]]               # Targets looking at the context
    dist2 = (xc[:, None, :] - xt[:, :, None]) ** 2          # (b, n_tgt, n_ctx)
    return jnp.sqrt((W * dist2[:, None, None]).sum(-1)).mean(-1)

tf_params = runs["transformer"][0]

# Pick the most local head in the last layer
xc, yc, xt, yt, _ = sample_tasks(jax.random.key(12), 300, 25, N_TGT)
widths = attention_width(attention_maps(tf_params, xc, yc, xt), xc, xt).mean(0)    # (layers, heads)
LAYER, HEAD = N_LAYERS - 1, int(jnp.argmin(widths[-1]))
print(f"attention width of every head (rows = layers):\n{np.array(widths).round(2)}")

# How the width of that head changes with the lengthscale and the number of context points
ells = np.geomspace(0.1, 3.0, 12)
width_vs_ell = {n_ctx: [] for n_ctx in [5, 10, 20, 40]}
for n_ctx in width_vs_ell:
    for ell_ in ells:
        xc, yc, xt, yt, _ = sample_tasks(jax.random.key(13), 300, n_ctx, N_TGT, ell=float(ell_))
        width_vs_ell[n_ctx].append(float(attention_width(attention_maps(tf_params, xc, yc, xt), xc, xt)[:, LAYER, HEAD].mean()))

# One function, and where every point on the grid looks
xc1, yc1, _, yt1, ell1 = sample_tasks(jax.random.key(14), 1, 15, ell=0.3, x_tgt=grid)
W1 = attention_maps(tf_params, xc1, yc1, grid[None])[0, LAYER, HEAD, 15:, :15]      # (grid, context)
mean1 = predict(transformer, tf_params, xc1, yc1, grid[None])[0][0]

blues = plt.matplotlib.colors.LinearSegmentedColormap.from_list("blues", ["#ffffff", colors["transformer"], "#0d2a5c"])
fig = plt.figure(figsize=(14, 6.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.2], width_ratios=[1.25, 1], hspace=0.08, wspace=0.32)
ax_fn = fig.add_subplot(gs[0, 0])
ax_map = fig.add_subplot(gs[1, 0], sharex=ax_fn)
ax_width = fig.add_subplot(gs[:, 1])

# Top: the function and the transformer's prediction
ax_fn.plot(np.array(grid), np.array(yt1[0]), color="#b8b8b4", lw=1)
ax_fn.plot(np.array(grid), np.array(mean1), color=colors["transformer"], lw=1.6)
ax_fn.scatter(np.array(xc1[0]), np.array(yc1[0]), s=16, color="#0b0b0b", zorder=3)
ax_fn.set_ylabel("y"); ax_fn.tick_params(labelbottom=False)
ax_fn.set_title(f"layer {LAYER + 1}, head {HEAD + 1}: where does the prediction at x look?", fontsize=11, loc="left")

# Bottom: one horizontal line per context point, at its x, darker where targets look at it
for j in range(15):
    ax_map.scatter(np.array(grid), np.full(grid.shape, float(xc1[0, j])), c=np.array(W1[:, j]), cmap=blues,
                   vmin=0, vmax=1, marker="s", s=6, lw=0)
ax_map.plot([-X_MAX, X_MAX], [-X_MAX, X_MAX], color="#0b0b0b", ls=":", lw=1, label="context point at the same x")
ax_map.legend(fontsize=9, loc="lower right")
ax_map.set_xlim(-X_MAX, X_MAX); ax_map.set_ylim(-X_MAX - 0.1, X_MAX + 0.1)
ax_map.set_xlabel("x of the target"); ax_map.set_ylabel("x of the context point")
sm = plt.cm.ScalarMappable(cmap=blues, norm=plt.Normalize(0, 1))
fig.colorbar(sm, ax=ax_map, pad=0.01, fraction=0.04, label="attention weight")

# Right: width of the attention vs. the lengthscale of the function
ax_width.axvspan(ELL_MIN, ELL_MAX, color="#f0f0ec", label="lengthscales seen in training")
shades = ["#a9c4f0", "#6f9be6", colors["transformer"], "#0d2a5c"]      # Light to dark = few to many points
for (n_ctx, w), shade in zip(width_vs_ell.items(), shades):
    ax_width.semilogx(ells, w, "o-", color=shade, lw=1.6, ms=4, label=f"{n_ctx} context points")
ax_width.set_xlabel("true lengthscale of the function"); ax_width.set_ylabel("attention width (in x)")
ax_width.set_title("smoother function → looks further, more points → looks closer", fontsize=11, loc="left")
ax_width.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1, 0.88))
plt.show()


'''

2. DOES IT KNOW WHAT IT DOESN'T KNOW?

The models also predict a variance, and since the data come from a GP we know the exact
right answer for that too. On the right, every dot is one target from 2000 test functions
(10 context points each): predicted std vs. the exact GP std. The transformer sits along the
diagonal: it is confident right next to a context point (std ~ the noise, 0.05) and unsure
in the gaps, by about the right amount. The MLP barely knows where it is: it says "not sure"
almost everywhere, even right on top of a data point.

Both are honest though: ~95% of targets land inside +/- 2 std for both (printed below). The
MLP is just honest by being vague.

The transformer is a bit LESS sure than the GP in the gaps (left panel, and above the
diagonal on the right). At least part of that is correct: the GP is told the lengthscale,
but the transformer has to guess it, so in a gap it also has to allow for the function
being wigglier than it looks.


'''

xc, yc, xt, yt, ell = sample_tasks(jax.random.key(15), 2000, 10, N_TGT)
preds = {}                                                                  # (mean, std) at every target
for name, (model, _) in models.items():
    mean, log_var = predict(model, runs[name][0], xc, yc, xt)
    preds[name] = (mean, jnp.exp(0.5 * log_var))
gp_mean, gp_cov = gp_posterior(xc, yc, xt, ell)
preds["GP"] = (gp_mean, jnp.sqrt(jnp.diagonal(gp_cov, axis1=1, axis2=2)))   # Exact posterior std

for name, (mean, std) in preds.items():
    inside = float((jnp.abs(yt - mean) < 2 * std).mean())
    print(f"{name:12s}: {inside:.1%} of targets inside +/- 2 std (should be 95.4%)")

# One function, transformer error bars on top of the exact ones
xc1, yc1, _, yt1, ell1 = sample_tasks(jax.random.key(16), 1, 8, x_tgt=grid)
gp_mean1, gp_cov1 = gp_posterior(xc1, yc1, grid[None], ell1)
gp_std1 = jnp.sqrt(jnp.diagonal(gp_cov1[0]))
tf_mean1, tf_log_var1 = predict(transformer, tf_params, xc1, yc1, grid[None])
tf_std1 = jnp.exp(0.5 * tf_log_var1[0])

fig, (ax_ex, *ax_hex) = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.5, 1, 1]})
g = np.array(grid)
ax_ex.fill_between(g, np.array(gp_mean1[0] - 2 * gp_std1), np.array(gp_mean1[0] + 2 * gp_std1), color="#e4e4e0", label="GP +/- 2 std (exact)")
ax_ex.plot(g, np.array(gp_mean1[0]), color="#0b0b0b", lw=1.4, ls="--", label="GP mean")
ax_ex.plot(g, np.array(tf_mean1[0]), color=colors["transformer"], lw=1.6, label="transformer mean")
for sign in [-1, 1]:
    ax_ex.plot(g, np.array(tf_mean1[0] + sign * 2 * tf_std1), color=colors["transformer"], lw=1, alpha=0.7)
ax_ex.plot([], [], color=colors["transformer"], lw=1, alpha=0.7, label="transformer +/- 2 std")
ax_ex.scatter(np.array(xc1[0]), np.array(yc1[0]), s=16, color="#0b0b0b", zorder=3)
ax_ex.set_xlabel("x"); ax_ex.set_ylabel("y"); ax_ex.legend(fontsize=9, loc="lower left")
ax_ex.set_title("8 context points", fontsize=11, loc="left")

# Every target of 2000 functions: predicted std vs. exact GP std
gp_std = preds["GP"][1]
for ax, name in zip(ax_hex, models):
    std = preds[name][1]
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(name, ["#ffffff", colors[name], "#0b0b0b"])
    ax.hexbin(np.array(gp_std).ravel(), np.array(std).ravel(), xscale="log", yscale="log", gridsize=45,
              bins="log", mincnt=1, cmap=cmap, extent=(np.log10(0.04), np.log10(1.5), np.log10(0.04), np.log10(1.5)))
    ax.plot([0.04, 1.5], [0.04, 1.5], color="#0b0b0b", ls=":", lw=1, label="perfect (same as GP)")
    corr = float(jnp.corrcoef(jnp.log(gp_std).ravel(), jnp.log(std).ravel())[0, 1])
    ax.set_title(f"{name}: correlation of log std = {corr:.2f}", fontsize=11, loc="left")
    ax.set_xlabel("exact GP std"); ax.set_ylabel(f"{name} std"); ax.legend(fontsize=9, loc="lower right")
plt.tight_layout(); plt.show()
