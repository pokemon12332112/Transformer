import jax
import optax
from flax.training import train_state
from flax import linen as nn
import numpy as np
from jax import random
import jax.numpy as jnp

class TrainState(train_state.TrainState):
    metrics: dict

def create_train_state(rng, learning_rate, model, src_vocab_size, tgt_vocab_size):
    params = model.init(rng, jnp.ones([1, 128], jnp.int32), jnp.ones([1, 128], jnp.int32))['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, metrics={})

def cross_entropy_loss(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1))

def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {'loss': loss, 'accuracy': accuracy}

@jax.jit
def train_step(state, batch, dropout_rng):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['src'], batch['tgt'], rngs={'dropout': dropout_rng})
        loss = cross_entropy_loss(logits, batch['tgt'])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch['tgt'])
    return state, metrics

    
@jax.jit
def eval_step(state, batch):
    logits = state.apply_fn({'params': state.params}, batch['src'], batch['tgt'])
    return compute_metrics(logits, batch['tgt'])

def convert_to_jax(batch):
    return {k: jnp.array(v) for k, v in batch.items()}

def train_epoch(state, train_loader, rng):
    batch_metrics = []
    for batch in train_loader:
        batch = convert_to_jax(batch)
        rng, input_rng = jax.random.split(rng)
        dropout_rng, input_rng = jax.random.split(input_rng)
        state, metrics = train_step(state, batch, dropout_rng)
        batch_metrics.append(metrics)
    
    train_metrics = {
        k: np.mean([metrics[k] for metrics in batch_metrics])
        for k in batch_metrics[0]
    }
    return state, train_metrics

def evaluate(state, val_loader):
    batch_metrics = []
    for batch in val_loader:
        batch = convert_to_jax(batch)
        metrics = eval_step(state, batch)
        batch_metrics.append(metrics)
    
    val_metrics = {
        k: np.mean([metrics[k] for metrics in batch_metrics])
        for k in batch_metrics[0]
    }
    return val_metrics