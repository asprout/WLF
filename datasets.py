import jax

from jax import numpy as jnp
import numpy as np

import scanpy as sc
import ot


# ---------------------------------------------------------------------------
# lfvae extension: manifold-aware coupling (interpolant='diffusion'|'geodesic').
#
# The main process (src/dynamics/manifold_coupling.py) precomputes the
# per-consecutive-pair log(plan) chain on a kNN graph of the latents and injects
# it here before training. ``get_batch_iterator`` then feeds it to the SAME
# ``ot_train_iterator`` the Euclidean 'ot' interpolant uses — only the source of
# the coupling changes (connectivity-respecting instead of Euclidean cost).
# ---------------------------------------------------------------------------
_PRECOMPUTED_LOG_PLANS = None


def set_precomputed_log_plans(log_plans):
    """Install the per-pair ``log(plan)`` chain used by the diffusion/geodesic
    interpolant. ``log_plans[i]`` is ``[n_i, n_{i+1}]`` aligning to consecutive
    ``X_train`` bins. Call before ``get_batch_iterator``."""
    global _PRECOMPUTED_LOG_PLANS
    _PRECOMPUTED_LOG_PLANS = [jnp.asarray(np.asarray(lp)) for lp in log_plans]


# ---------------------------------------------------------------------------
# lfvae extension: optional per-cell weights for sink supervision.
#
# When ``config.train.sink_weight > 0``, the training iterator yields a
# 3-tuple ``(t_batch, x_batch, w_batch)``; otherwise the original 2-tuple
# ``(t_batch, x_batch)`` is yielded for full backwards compatibility with
# every existing wl-mechanics loss / config.
#
# The weights flow through TWO seams:
#
#   1. ``get_data`` may return a 6-tuple
#         (X_train, X_test, X_val, inv_scaler, times, W_train)
#      where ``W_train`` is a list of per-bin numpy arrays (n_i, 1) of
#      potency-derived sink weights aligned cell-for-cell with ``X_train[i]``.
#      The 5-tuple legacy form remains supported — when get_data returns 5
#      elements we synthesise ``W_train = None`` and emit the 2-tuple
#      batches (no sink supervision possible without per-cell weights).
#
#   2. The downstream worker (``tools/wlf_import_worker.py``) is the only
#      caller that wants sink supervision in our pipeline. It monkeypatches
#      ``get_data`` to return per-bin latents AND per-bin weights (potency
#      threshold derived from the AnnData), so the seam is satisfied without
#      touching any upstream config dispatch.
#
# Implementation note: we DON'T touch ``ot_train_iterator``. The OT-coupling
# branch precomputes EMD plans between consecutive marginals and re-samples
# cells from the joint plan distribution; threading per-cell weights through
# an EMD plan is not well-defined (a plan re-shuffles row identities). The
# linear ('independent coupling') branch is what we use with sink-supervised
# RF, and is the natural fit — each cell-bin draw is i.i.d., so its weight
# travels with it.
# ---------------------------------------------------------------------------


def _unpack_get_data(ret):
  """Tolerate both the 5-tuple legacy return shape and the 6-tuple
  weights-aware return shape from ``get_data``. Returns a 6-tuple with
  ``W_train`` defaulted to None when absent."""
  if len(ret) == 6:
    X_train, X_test, X_val, inv_scaler, times, W_train = ret
  elif len(ret) == 5:
    X_train, X_test, X_val, inv_scaler, times = ret
    W_train = None
  else:
    raise ValueError(
      f"get_data() must return 5 or 6 elements, got {len(ret)}.")
  return X_train, X_test, X_val, inv_scaler, times, W_train


def get_batch_iterator(config, init_key, eval=False, val=False):
  batch_size = config.eval.batch_size if eval else config.train.batch_size
  X_train, X_test, X_val, inv_scaler, times, W_train = _unpack_get_data(
    get_data(config, init_key))
  assert len(X_train) == len(X_test)
  print(times, 'times', flush=True)
  t = ((times - np.min(times))/(np.max(times)-np.min(times))).tolist()
  print(t, 'times normalized', flush=True)
  # Sink supervision active iff the loss requested it (sink_weight > 0) AND
  # the data source provided per-cell weights. Inform loudly so a misconfig
  # (sink_weight>0 but data has no weights) doesn't silently degrade to
  # baseline.
  _sink_weight = float(getattr(config.train, 'sink_weight', 0.0))
  yield_weights = bool(W_train is not None) and (_sink_weight > 0.0) and (not eval)
  if _sink_weight > 0.0 and W_train is None and not eval:
    print(
      "[datasets] WARNING: config.train.sink_weight > 0 but get_data() "
      "returned no per-cell weights; sink supervision will be DISABLED. "
      "Use the wlf_import_worker data path or extend get_data to emit a "
      "6th element (W_train).", flush=True)
    
  if config.data.name == '4i':
    def test_iterator():
      return ([X_test[0]], [t[0]], [X_test[-1]], [t[-1]])
    def val_iterator():
      return ([X_val[0]], [t[0]], [X_val[-1]], [t[-1]])
  elif config.data.test_id is not None:
    assert config.data.test_id < (len(X_train)-1) and config.data.test_id > 0
    def test_iterator():
      return ([X_train[config.data.test_id-1]], [t[config.data.test_id-1]], 
              [X_train.pop(config.data.test_id)], [t.pop(config.data.test_id)])
    def val_iterator():
      return ([X_train[config.data.test_id-1]], [t[config.data.test_id-1]], 
              [X_train.pop(config.data.test_id)], [t.pop(config.data.test_id)])
  else: 
    def test_iterator():
      return (X_test[:-1], t[:-1], X_test[1:], t[1:])
    def val_iterator():
      return (X_val[:-1], t[:-1], X_val[1:], t[1:])
  if eval:
    if val:
      return val_iterator, inv_scaler
    else:
      return test_iterator, inv_scaler
    
  # Pre-convert X_train AND (if present) W_train to jnp arrays BEFORE the
  # @jax.jit'd iterator body. The original upstream iterator could get away
  # with numpy X_train because it used the population-array form
  # ``jax.random.choice(key, X_train[i], (B,))`` which dispatches the
  # numpy→jax conversion internally. We need to gather indices SEPARATELY so
  # the same indices can index both x and w (sink alignment), and that
  # ``X_train[i][ids]`` pattern requires X_train[i] to already be a jnp
  # array — otherwise a numpy.__array__() call on a traced index fires at
  # jit-trace time. (Bug fix: previously crashed every linear-interpolant
  # variant with ``TracerArrayConversionError``.)
  X_train_jnp = [jnp.asarray(np.asarray(arr, dtype=np.float32))
                 for arr in X_train]
  if yield_weights:
    W_jnp = []
    for i, w in enumerate(W_train):
      w_arr = np.asarray(w, dtype=np.float32).reshape(-1, 1)
      if w_arr.shape[0] != X_train_jnp[i].shape[0]:
        raise ValueError(
          f"W_train[{i}] length {w_arr.shape[0]} != X_train[{i}] length "
          f"{X_train_jnp[i].shape[0]}; weights must align cell-for-cell.")
      W_jnp.append(jnp.asarray(w_arr))

  @jax.jit
  def linear_train_iterator(key):
    keys = jax.random.split(key, len(X_train_jnp))
    x_batch = jnp.zeros((batch_size, len(X_train_jnp), config.data.dim))
    t_batch = jnp.zeros((batch_size, len(X_train_jnp), 1))
    if yield_weights:
      w_batch = jnp.zeros((batch_size, len(X_train_jnp), 1))
    for i in range(len(X_train_jnp)):
      # Sample cell indices once, use the same indices to gather both x and w
      # so weights stay aligned with the cells they belong to.
      ids = jax.random.choice(
        keys[i], X_train_jnp[i].shape[0], (batch_size,), replace=True)
      x_batch = x_batch.at[:, i, :].set(X_train_jnp[i][ids])
      t_batch = t_batch.at[:, i, :].set(t[i])
      if yield_weights:
        w_batch = w_batch.at[:, i, :].set(W_jnp[i][ids])
    x_batch = x_batch.reshape(jax.local_device_count(),
                              config.train.n_jitted_steps,
                              batch_size // jax.local_device_count(),
                              len(X_train),
                              config.data.dim)
    t_batch = t_batch.reshape(jax.local_device_count(),
                              config.train.n_jitted_steps,
                              batch_size // jax.local_device_count(),
                              len(t),
                              1)
    if yield_weights:
      w_batch = w_batch.reshape(jax.local_device_count(),
                                config.train.n_jitted_steps,
                                batch_size // jax.local_device_count(),
                                len(t),
                                1)
      return (t_batch, x_batch, w_batch)
    return (t_batch, x_batch)

  if config.interpolant == 'linear':
    return linear_train_iterator, inv_scaler
  
  # ot_train_iterator does NOT support sink supervision: the categorical-
  # sampling from the joint plan re-shuffles cell identities across the
  # consecutive pair, so per-cell weights would no longer align with the
  # cells they were derived from. Warn loudly if the caller asked for both.
  if yield_weights and config.interpolant in ('ot', 'diffusion', 'geodesic'):
    print(
      "[datasets] WARNING: sink_weight > 0 with interpolant='%s' is not "
      "supported; sink supervision will be DISABLED. Use "
      "interpolant='linear' to enable sink supervision." % config.interpolant,
      flush=True)
    yield_weights = False

  if config.interpolant in ('diffusion', 'geodesic'):
    # Manifold-aware coupling: use the precomputed plan chain (set via
    # set_precomputed_log_plans) instead of the Euclidean EMD plan.
    if _PRECOMPUTED_LOG_PLANS is None:
      raise RuntimeError(
        f"interpolant={config.interpolant!r} requires "
        f"datasets.set_precomputed_log_plans(...) to be called first.")
    log_plans = list(_PRECOMPUTED_LOG_PLANS)
    if len(log_plans) != len(X_train) - 1:
      raise ValueError(
        f"precomputed log_plans has {len(log_plans)} entries; expected "
        f"{len(X_train) - 1} for {len(X_train)} bins.")
    for i in range(len(log_plans)):
      exp = (X_train[i].shape[0], X_train[i + 1].shape[0])
      if tuple(log_plans[i].shape) != exp:
        raise ValueError(
          f"precomputed log_plan {i} shape {tuple(log_plans[i].shape)} != "
          f"({exp[0]}, {exp[1]}) from X_train bins.")
  else:
    log_plans = []
    for i in range(len(X_train)-1):
      a, b = ot.unif(X_train[i].shape[0]), ot.unif(X_train[i+1].shape[0])
      M = ot.dist(X_train[i], X_train[i+1], metric='euclidean')
      plan = ot.emd(np.array(a).astype(np.float32),
                    np.array(b).astype(np.float32),
                    np.array(M).astype(np.float32),numItermax=1e7)
      plan = plan/plan.sum(1, keepdims=True)
      log_plans.append(jnp.array(np.log(plan)))
    
  for i in range(len(X_train)):
    X_train[i] = jnp.array(X_train[i])
  
  @jax.jit
  def ot_train_iterator(key):
    keys = jax.random.split(key, len(X_train))
    x_batch = jnp.zeros((batch_size, len(X_train), config.data.dim))
    t_batch = jnp.zeros((batch_size, len(X_train), 1))
    for i in range(len(X_train)):
      if i == 0:
        ids = jax.random.categorical(keys[i], np.zeros((X_train[0].shape[0],)), shape=(batch_size,))
      else:
        ids = jax.random.categorical(keys[i], log_plans[i-1][ids], axis=1, shape=(batch_size,))
      x_batch = x_batch.at[:,i,:].set(X_train[i][ids])
      t_batch = t_batch.at[:,i,:].set(t[i])
    
    x_batch = x_batch.reshape(jax.local_device_count(),
                              config.train.n_jitted_steps, 
                              batch_size//jax.local_device_count(),
                              len(X_train),
                              config.data.dim)
    t_batch = t_batch.reshape(jax.local_device_count(),
                              config.train.n_jitted_steps, 
                              batch_size//jax.local_device_count(),
                              len(t),
                              1)
    return (t_batch, x_batch)
  
  if config.interpolant in ('ot', 'diffusion', 'geodesic'):
    return ot_train_iterator, inv_scaler

  raise NotImplementedError(f'{config.interpolant} is not implemented')

def get_data(config, init_key):
  if config.data.name == '4i':
    return get_rna_data(config, init_key)
  if config.data.name == 'rna':
    return get_rna_data(config, init_key)
  if config.data.name == 'toy':
    return get_toy_data(config, init_key)
  if config.data.name == 'embrio':
    return get_h5ad_data(config, init_key)
  if config.data.name == 'cite':
    return get_h5ad_data(config, init_key)
  if config.data.name == 'multi':
    return get_h5ad_data(config, init_key)
  NotImplementedError(f'config.data.name: {config.data.name} is not implemented')
  

def get_h5ad_data(config, init_key):
  if config.data.name == 'embrio':
    adata = sc.read_h5ad("assets/ebdata_v3.h5ad")
    adata.obs["day"] = adata.obs["sample_labels"].cat.codes
  elif config.data.name == 'cite':
    adata = sc.read_h5ad("assets/op_cite_inputs_0.h5ad")
  elif config.data.name == 'multi':
    adata = sc.read_h5ad("assets/op_train_multi_targets_0.h5ad")
  else:
    NotImplementedError(f'config.data.name: {config.data.name} is not implemented')
  times = adata.obs["day"].unique()
  coords = adata.obsm["X_pca"][:,:config.data.dim]
  if config.data.whiten:
    mu = coords.mean(axis=0, keepdims=True)
    sigma = coords.std(axis=0, keepdims=True)
    coords = (coords - mu) / sigma
    inv_scaler = lambda _x: _x
  else:
    mu = coords.mean(axis=0, keepdims=True)
    sigma = np.max(coords.std(axis=0, keepdims=True))
    coords = (coords - mu) / sigma
    inv_scaler = lambda _x: _x*sigma + mu
  adata.obsm["X_pca_standardized"] = coords
  X = [
    adata.obsm["X_pca_standardized"][adata.obs["day"] == t]
    for t in times
  ]
  X_train, X_test, X_val = X, X, X
  times = np.linspace(0.0, 1.0, len(X)).tolist()
  return X_train, X_test, X_val, inv_scaler, times


def get_rna_data(config, init_key):
  def load_rna(filename):
    with np.load(filename) as data:
      t = data['ts']
      X = data['X'][:,:config.data.dim]
    return t, X
  t_train, X_train = load_rna(f'assets/train_{config.data.name}.npz')
  t_test, X_test = load_rna(f'assets/test_{config.data.name}.npz')
  t_val, X_val = load_rna(f'assets/val_{config.data.name}.npz')
  if config.data.whiten:
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    X_val = (X_val - mu) / sigma
    inv_scaler = lambda _x: _x
  else:
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = np.max(X_train.std(axis=0, keepdims=True))
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    X_val = (X_val - mu) / sigma
    inv_scaler = lambda _x: _x*sigma + mu
    
  times = np.unique(t_train)
  X_train = [jnp.array(X_train[t_train == t]) for t in times]
  X_test = [jnp.array(X_test[t_test == t]) for t in times]
  X_val = [jnp.array(X_val[t_val == t]) for t in times]
  return X_train, X_test, X_val, inv_scaler, times


def get_toy_data(config, init_key):
  if config.loss == 'am':
    return get_toy_data_for_ot(config, init_key)
  if config.loss == 'sb':
    return get_toy_data_for_ot(config, init_key)
  if config.loss == 'ubot':
    return get_toy_data_for_ubot(config, init_key)
  if config.loss == 'phot':
    return get_toy_data_for_phot(config, init_key)

def get_toy_data_for_ubot(config, init_key):
  init_key = jax.random.split(init_key)
  DS = 10_000
  sigma = 3e-1
  X_init = jnp.concatenate([-jnp.ones((DS,1)), jnp.zeros((DS,1))], 1)
  X_final = -X_init
  X_init += sigma*jax.random.normal(init_key[0], shape=(X_init.shape[0], 2))
  X_final += sigma*jax.random.normal(init_key[1], shape=(X_final.shape[0], 2))
  
  X_train = X_val = X_test = [X_init, X_final]
  inv_scaler = lambda _x: _x
  times = np.array([0.0, 1.0])
  return  X_train, X_test, X_val, inv_scaler, times

def get_toy_data_for_phot(config, init_key):
  init_key = jax.random.split(init_key)
  DS = 10_000
  sigma = 1e-1
  X_init = jnp.concatenate([-jnp.ones((DS,1)), jnp.zeros((DS,1))], 1)
  X_final = -X_init
  X_init += sigma*jax.random.normal(init_key[0], shape=(DS, 2))
  X_final += sigma*jax.random.normal(init_key[1], shape=(DS, 2))
  
  X_train = X_val = X_test = [X_init, X_final]
  inv_scaler = lambda _x: _x
  times = np.array([0.0, 1.0])
  return  X_train, X_test, X_val, inv_scaler, times

def get_toy_data_for_ot(config, init_key):
  init_key = jax.random.split(init_key)
  DS = 10_000
  sigma = 1e-1
  X_init = (jnp.ones((DS//8, 8))*(2*jnp.pi*jnp.arange(8)/8).reshape(1,-1)).reshape(-1,1)
  X_init = jnp.concatenate([jnp.cos(X_init), jnp.sin(X_init)], 1)
  X_final = 2*X_init
  X_init += sigma*jax.random.normal(init_key[0], shape=(DS, 2))
  X_final += sigma*jax.random.normal(init_key[1], shape=(DS, 2))
  X_train = X_val = X_test = [X_init, X_final]
  inv_scaler = lambda _x: _x
  times = np.array([0.0, 1.0])
  return  X_train, X_test, X_val, inv_scaler, times
