# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions and modules related to model definition.
"""
from typing import Any

import flax
import optax
import functools
import jax.numpy as jnp
import jax
import numpy as np
# from utils import batch_mul


# The dataclass that stores all training states
@flax.struct.dataclass
class State:
  step: int
  opt_state: Any
  model_params: Any
  ema_rate: float
  params_ema: Any
  key: Any
  sampler_state: Any
  wandbid: Any


_MODELS = {}


def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]


class HelmholtzModelPair:
  """Combine a scalar potential ``U_θ(t, x)`` and a curl ``c_ψ(t, x)`` into a
  single object whose ``.apply()`` returns the Helmholtz velocity

      v(t, x) = +∇_x U_θ(t, x) + c_ψ(t, x).

  Mirrors ``flax.linen.Module``'s ``apply`` interface (variables-dict in,
  array out) so it is a **drop-in replacement** wherever the codebase
  currently accepts a single flax Module for ``model_s`` — in particular
  ``mutils.get_model_fn`` and ``eval_utils.get_generator``. This is a plain
  Python class (NOT a flax Module) on purpose: we **need** the nested-params
  structure ``{"potential": ..., "curl": ...}`` so the JAX value-and-grad
  inside ``train_utils.get_step_fn`` differentiates **both** heads with no
  changes to the gradient pipeline.

  Sign convention follows wl-mechanics (action S, ``v = +∇S``) — see
  ``MlpCurl``'s docstring for the relationship to the spec's energy
  ``E = −U``.

  In addition to ``apply``, two convenience accessors expose each head
  separately, used by the divergence / curl-L2 regularisers and by the
  torch-twin equivalence test:

      .apply_potential(...)  → U(t, x)  : (B, 1) scalar
      .apply_curl(...)       → c(t, x)  : (B, D) vector
  """

  def __init__(self, potential_model, curl_model):
    self.potential = potential_model
    self.curl = curl_model

  def _unpack(self, variables):
    # Tolerate either ``{"params": {...}}`` (flax convention used by
    # get_model_fn) or a raw nested dict, so callers don't have to remember
    # which form they hold.
    params = variables["params"] if "params" in variables else variables
    return params["potential"], params["curl"]

  def apply(self, variables, t, x, train=False, mutable=False, rngs=None):
    """Return the Helmholtz velocity v = +∇_x U(t, x) + c(t, x)."""
    pot_p, curl_p = self._unpack(variables)
    # Both subapplies share the same dropout rng — acceptable because the
    # two heads are structurally independent (no shared parameters) and
    # dropout is on by default zero (config.dropout=0.0 in our configs).
    rngs_u = rngs
    rngs_c = rngs

    c = self.curl.apply({"params": curl_p}, t, x, train=train,
                        mutable=mutable, rngs=rngs_c)

    def _u_sum(_x):
      return self.potential.apply({"params": pot_p}, t, _x, train=train,
                                   mutable=mutable, rngs=rngs_u).sum()
    grad_u = jax.grad(_u_sum)(x)
    return grad_u + c

  def apply_potential(self, variables, t, x, train=False, mutable=False,
                      rngs=None):
    pot_p, _ = self._unpack(variables)
    return self.potential.apply({"params": pot_p}, t, x, train=train,
                                 mutable=mutable, rngs=rngs)

  def apply_curl(self, variables, t, x, train=False, mutable=False,
                 rngs=None):
    _, curl_p = self._unpack(variables)
    return self.curl.apply({"params": curl_p}, t, x, train=train,
                            mutable=mutable, rngs=rngs)


def _init_single_model_s(rng, config):
  """Build a single scalar-output (or vector-output, for ``mlp_vf``)
  ``flax.linen.Module`` from ``config.name`` and return ``(model, params)``.
  This is the original ``init_model_s`` body, factored out so the Helmholtz
  branch can reuse it for both the potential and the curl head."""
  model_name = config.name
  model_def = functools.partial(get_model(model_name), config=config)
  x_shape = (jax.local_device_count(), config.input_dim)
  t_shape = (jax.local_device_count(), 1)
  fake_x = jnp.zeros(x_shape)
  fake_t = jnp.zeros(t_shape, dtype=jnp.int32)
  params_rng, dropout_rng = jax.random.split(rng)
  model = model_def()
  variables = model.init({'params': params_rng, 'dropout': dropout_rng},
                          fake_t, fake_x, train=True)
  initial_params = variables.pop('params')
  return model, initial_params


def init_model_s(rng, config):
  """Initialise ``model_s``. Two modes:

    - **default** (``config.helmholtz`` absent or False): the original
      behaviour — build one ``flax.linen.Module`` from ``config.name``,
      return ``(model, params)``.
    - **Helmholtz** (``config.helmholtz`` is True, lfvae extension): build a
      potential ``U_θ`` from ``config.name`` (default ``"mlp_scalar_s"`` per
      the config template) and a curl ``c_ψ`` from
      ``config.curl_name`` (default ``"mlp_curl"`` — zero-init final layer).
      Return ``(HelmholtzModelPair, {"potential": ..., "curl": ...})``.

  The Helmholtz pair's ``.apply()`` returns the velocity directly, so the
  rest of the training stack (``mutils.get_model_fn``, ``losses.get_loss``,
  ``eval_utils.get_generator``) treats it like a vector-field model (``mlp_vf``)
  — no caller-side branching beyond a ``loss == 'rf' or helmholtz`` check in
  ``eval_utils.get_generator`` to skip the now-irrelevant ``jax.grad(s)``.
  """
  if not bool(getattr(config, "helmholtz", False)):
    return _init_single_model_s(rng, config)

  # Helmholtz branch.
  rng_u, rng_c = jax.random.split(rng)
  potential_model, potential_params = _init_single_model_s(rng_u, config)

  # Curl-head config: share trunk hyperparameters (nf, n_layers, embed_time,
  # nonlinearity, dropout, skip) with the potential, but force the output-
  # shape-specific bits — name='mlp_curl' (vector output + zero-init) and
  # output_dim=input_dim (already true; carried through).
  import ml_collections
  curl_config = ml_collections.ConfigDict(config.to_dict())
  curl_config.name = str(getattr(config, "curl_name", "mlp_curl"))
  curl_model, curl_params = _init_single_model_s(rng_c, curl_config)

  pair = HelmholtzModelPair(potential_model, curl_model)
  return pair, {"potential": potential_params, "curl": curl_params}


def init_model_q(rng, config):
  """ Initialize a `flax.linen.Module` model. """
  model_name = config.name
  model_def = functools.partial(get_model(model_name), config=config)
  timesteps_shape = (jax.local_device_count(), config.n_marginals, 1)
  x_shape = (jax.local_device_count(), config.n_marginals, config.input_dim)
  t_shape = (jax.local_device_count(), 1)
  fake_batch = (jnp.zeros(timesteps_shape), jnp.zeros(x_shape))
  fake_t = jnp.zeros(t_shape, dtype=jnp.int32)
  params_rng, dropout_rng = jax.random.split(rng)
  model = model_def()
  variables = model.init({'params': params_rng, 'dropout': dropout_rng}, fake_t, fake_batch, train=True)
  # Variables is a `flax.FrozenDict`. It is immutable and respects functional programming
  initial_params = variables.pop('params')
  return model, initial_params


def get_model_fn(model, params, train=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: A `flax.linen.Module` object the represent the architecture of score-based model.
    params: A dictionary that contains all trainable parameters.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(t, x, rng=None):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.
      rng: If present, it is the random state for dropout

    Returns:
      A tuple of (model output, new mutable states)
    """
    variables = dict(params=params)
    if not train:
      return model.apply(variables, t, x, train=False, mutable=False)
    else:
      rngs = {'dropout': rng}
      return model.apply(variables, t, x, train=True, mutable=False, rngs=rngs)

  return model_fn


def to_flattened_numpy(x):
  """Flatten a JAX array `x` and convert it to numpy."""
  return np.asarray(x.reshape((-1,)))


def from_flattened_numpy(x, shape):
  """Form a JAX array with the given `shape` from a flattened numpy array `x`."""
  return jnp.asarray(x).reshape(shape)
