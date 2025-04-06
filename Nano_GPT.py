import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import train_state
import optax

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200
n_embd = 24
n_head = 6
n_layer = 6
dropout_rate = 0.2  

key = jax.random.PRNGKey(1337)

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
    data_arr = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(data_arr) - block_size, size=(batch_size,))
    x = np.stack([data_arr[i : i + block_size] for i in ix])
    y = np.stack([data_arr[i + 1 : i + block_size + 1] for i in ix])
    return jnp.array(x), jnp.array(y)


class Head(nn.Module):
    head_size: int
    block_size: int
    dropout_rate: float

    def setup(self):
        self.key = nn.Dense(self.head_size, use_bias=False)
        self.query = nn.Dense(self.head_size, use_bias=False)
        self.value = nn.Dense(self.head_size, use_bias=False)
        self.tril = jnp.tril(jnp.ones((self.block_size, self.block_size)))
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, deterministic: bool):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        wei = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * (self.head_size ** -0.5)
        wei = jnp.where(self.tril[:T, :T] == 0, -1e10, wei)
        wei = nn.softmax(wei, axis=-1)
        wei = self.dropout(wei, deterministic=deterministic)
        out = jnp.matmul(wei, v)
        return out

class MultiHeadAttention(nn.Module):
    num_heads: int
    n_embd: int
    block_size: int
    dropout_rate: float

    def setup(self):
        head_size = self.n_embd // self.num_heads
        self.heads = [Head(head_size, self.block_size, self.dropout_rate) for _ in range(self.num_heads)]
        self.proj = nn.Dense(self.n_embd)
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, deterministic: bool):
        head_outputs = [head(x, deterministic) for head in self.heads]
        out = jnp.concatenate(head_outputs, axis=-1)
        out = self.proj(out)
        out = self.dropout(out, deterministic=deterministic)
        return out

class FeedForward(nn.Module):
    n_embd: int
    dropout_rate: float

    def setup(self):
        self.dense1 = nn.Dense(4 * self.n_embd)
        self.dense2 = nn.Dense(self.n_embd)
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, deterministic: bool):
        x = self.dense1(x)
        x = nn.relu(x)
        x = self.dense2(x)
        x = self.dropout(x, deterministic=deterministic)
        return x

class Block(nn.Module):
    n_embd: int
    n_head: int
    block_size: int
    dropout_rate: float

    def setup(self):
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        self.sa = MultiHeadAttention(self.n_head, self.n_embd, self.block_size, self.dropout_rate)
        self.ffwd = FeedForward(self.n_embd, self.dropout_rate)

    def __call__(self, x, deterministic: bool):
        x = x + self.sa(self.ln1(x), deterministic=deterministic)
        x = x + self.ffwd(self.ln2(x), deterministic=deterministic)
        return x

class BigramLanguageModel(nn.Module):
    vocab_size: int
    n_embd: int
    block_size: int
    n_layer: int
    n_head: int
    dropout_rate: float

    def setup(self):
        self.token_embedding_table = nn.Embed(num_embeddings=self.vocab_size, features=self.n_embd)
        self.position_embedding_table = nn.Embed(num_embeddings=self.block_size, features=self.n_embd)
        self.blocks = [Block(self.n_embd, self.n_head, self.block_size, self.dropout_rate) for _ in range(self.n_layer)]
        self.ln_f = nn.LayerNorm()
        self.lm_head = nn.Dense(self.vocab_size)

    def __call__(self, idx, targets=None, deterministic: bool = False):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(jnp.arange(T))
        x = tok_emb + pos_emb  
        for block in self.blocks:
            x = block(x, deterministic=deterministic)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            logits_flat = logits.reshape(-1, self.vocab_size)
            targets_flat = targets.reshape(-1)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, targets_flat).mean()
        return logits, loss

    def generate(self, params, rng, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.apply({'params': params}, idx_cond, deterministic=True)
            logits_last = logits[:, -1, :]  # (B, vocab_size)
            probs = nn.softmax(logits_last)
            rng, subkey = jax.random.split(rng)
            next_token = jax.random.categorical(subkey, jnp.log(probs), axis=-1)
            next_token = next_token[:, None]  # (B, 1)
            idx = jnp.concatenate([idx, next_token], axis=1)
        return idx, rng


class TrainState(train_state.TrainState):
    pass

model = BigramLanguageModel(vocab_size, n_embd, block_size, n_layer, n_head, dropout_rate)
rng, init_rng = jax.random.split(key)
dropout_rng = jax.random.PRNGKey(42)
dummy_input = jnp.ones((batch_size, block_size), dtype=jnp.int32)
params = model.init({'params': init_rng, 'dropout': dropout_rng}, dummy_input, deterministic=False)['params']

tx = optax.adamw(learning_rate)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch, dropout_rng):
    def loss_fn(params):
        _, loss = model.apply({'params': params}, batch[0], targets=batch[1], deterministic=False,
                                rngs={'dropout': dropout_rng})
        return loss
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def evaluate(state):
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch("val")
        _, loss = model.apply({'params': state.params}, x, targets=y, deterministic=True)
        losses.append(loss)
    return np.mean(np.array(losses))


for iter in range(max_iters):
    if iter % eval_interval == 0:
        train_losses = []
        val_loss = evaluate(state)
        for _ in range(eval_iters):
            x, y = get_batch("train")
            _, loss = model.apply({'params': state.params}, x, targets=y, deterministic=True)
            train_losses.append(loss)
        print(f"Step {iter}: train loss {np.mean(np.array(train_losses)):.4f}, val loss {val_loss:.4f}")
    x, y = get_batch("train")
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
    state, loss = train_step(state, (x, y), dropout_rng)

context = jnp.zeros((1, 1), dtype=jnp.int32)
generated, _ = model.generate(state.params, dropout_rng, context, max_new_tokens=500)
generated_text = decode(np.array(generated[0]))
print(generated_text)
