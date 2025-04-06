import jax
import jax.numpy as jnp
import flax.linen as nnx
import optax
import numpy as np
import functools
import random

# ------------------------
# Hyperparameters
# ------------------------
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200
n_embd = 24
n_head = 6
n_layer = 6
dropout = 0.2
key = jax.random.PRNGKey(1337)

# ------------------------
# Load data
# ------------------------
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = np.array(encode(text), dtype=np.int32)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(d) - block_size, size=(batch_size,))
    x = np.stack([d[i:i + block_size] for i in ix])
    y = np.stack([d[i + 1:i + block_size + 1] for i in ix])
    return jnp.array(x), jnp.array(y)

def causal_attention_mask(seq_len):
    return jnp.tril(jnp.ones((seq_len, seq_len)))

class Head(nnx.Module):
    head_size: int
    dropout: float

    @nnx.compact
    def __call__(self, x, train: bool, mask):
        k = nnx.Dense(self.head_size, use_bias=False)(x)
        q = nnx.Dense(self.head_size, use_bias=False)(x)
        v = nnx.Dense(self.head_size, use_bias=False)(x)

        att = jnp.einsum('bte,bse->bts', q, k) / jnp.sqrt(self.head_size)
        att = jnp.where(mask, att, -1e10)
        att = nnx.softmax(att, axis=-1)
        att = nnx.Dropout(rate=self.dropout)(att, deterministic=not train)
        out = jnp.einsum('bij,bjd->bid', att, v)
        return out

class MultiHeadAttention(nnx.Module):
    n_head: int
    head_size: int
    dropout: float

    @nnx.compact
    def __call__(self, x, train: bool, mask):
        heads = [Head(self.head_size, self.dropout)(x, train, mask) for _ in range(self.n_head)]
        out = jnp.concatenate(heads, axis=-1)
        out = nnx.Dense(self.n_head * self.head_size)(out)
        out = nnx.Dropout(rate=self.dropout)(out, deterministic=not train)
        return out

class FeedForward(nnx.Module):
    n_embd: int
    dropout: float

    @nnx.compact
    def __call__(self, x, train: bool):
        x = nnx.Dense(4 * self.n_embd)(x)
        x = nnx.relu(x)
        x = nnx.Dense(self.n_embd)(x)
        x = nnx.Dropout(rate=self.dropout)(x, deterministic=not train)
        return x

class Block(nnx.Module):
    n_embd: int
    n_head: int
    dropout: float

    @nnx.compact
    def __call__(self, x, train: bool, mask):
        x = x + MultiHeadAttention(self.n_head, self.n_embd // self.n_head, self.dropout)(nnx.LayerNorm()(x), train, mask)
        x = x + FeedForward(self.n_embd, self.dropout)(nnx.LayerNorm()(x), train)
        return x

class BigramLanguageModel(nnx.Module):
    vocab_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float

    @nnx.compact
    def __call__(self, idx, targets=None, train: bool = False):
        B, T = idx.shape
        tok_emb = nnx.Embed(self.vocab_size, self.n_embd)(idx)
        pos_emb = self.param('pos_emb', jax.nn.initializers.normal(), (block_size, self.n_embd))
        x = tok_emb + pos_emb[:T]

        mask = causal_attention_mask(T)[None, :, :]

        for _ in range(self.n_layer):
            x = Block(self.n_embd, self.n_head, self.dropout)(x, train, mask)

        x = nnx.LayerNorm()(x)
        logits = nnx.Dense(self.vocab_size)(x)

        loss = None
        if targets is not None:
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

        return logits, loss



# ------------------------
# Training
# ------------------------

model = BigramLanguageModel(vocab_size, n_embd, n_head, n_layer, dropout)
variables = model.init(key, jnp.ones((1, 1), dtype=jnp.int32), train=False)
params = variables['params']
tx = optax.adamw(learning_rate)
opt_state = tx.init(params)

@functools.partial(jax.jit, static_argnums=(3,))
def train_step(params, opt_state, batch, train):
    def loss_fn(p):
        logits, loss = model.apply({'params': p}, batch[0], batch[1], train=train)
        return loss, logits

    grads, _ = jax.grad(loss_fn, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

@jax.jit
def eval_step(params, batch):
    _, loss = model.apply({'params': params}, batch[0], batch[1], train=False)
    return loss

def estimate_loss(params):
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            loss = eval_step(params, (xb, yb))
            losses.append(loss)
        out[split] = np.mean(jnp.array(losses))
    return out

# ------------------------
# Training loop
# ------------------------

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss(params)
        print(f"Step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")
    params, opt_state = train_step(params, opt_state, (xb, yb), train=True)

# ------------------------
# Text generation
# ------------------------

def generate(params, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model.apply({'params': params}, idx_cond, train=False)
        logits = logits[:, -1, :]
        probs = jax.nn.softmax(logits, axis=-1)
        next_token = jax.random.categorical(jax.random.PRNGKey(random.randint(0, 1e6)), logits)
        idx = jnp.concatenate([idx, next_token[:, None]], axis=1)
    return idx

context = jnp.zeros((1, 1), dtype=jnp.int32)
generated = generate(params, context, max_new_tokens=500)
print(decode(np.array(generated[0])))
