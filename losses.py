import math

import flax
import jax
import jax.numpy as jnp
import jax.random as random
from models import utils as mutils


# ---------------------------------------------------------------------------
# lfvae Helmholtz / sink-supervision extensions.
#
# WHAT'S NEW (gated by config knobs; no-op when all are at their defaults):
#
#   config.model_s.helmholtz : bool           default False
#     When True, ``init_model_s`` builds a HelmholtzModelPair whose ``.apply``
#     returns velocity v = +∇U + c (see models/mlp.py:MlpCurl and
#     models/utils.py:HelmholtzModelPair). Currently supported ONLY by the
#     'rf' (rectified-flow / flow-matching) loss. With other losses, AM-
#     family in particular, the HJB-style action identification ``v = ∇S``
#     no longer holds when c ≠ 0 — adding a non-gradient component would
#     turn the loss into a poorly-conditioned objective whose stationary
#     point is c → 0 (we worked this through in the implementation plan).
#     We raise loudly rather than silently train a broken loss; the natural
#     home for Helmholtz here is RF, which already supervises v directly.
#
#   config.train.sink_weight : float          default 0.0
#     Per-cell sink supervision term added to the loss:
#         L_sink = (Σ w · ‖v(x_obs, t_obs)‖²) / Σ w
#     evaluated at observed cells (NOT the bridge-sampled x_t), with potency-
#     derived weights w threaded through the batch tuple (see datasets.py).
#     Normalised by Σw so the knob is dataset-invariant — without
#     normalisation, sink_weight would need re-tuning per batch composition
#     (high-progenitor batches vs low-progenitor batches scale Σw very
#     differently). Loss-agnostic: works in every loss family — for action-
#     style losses (am/sb/ubot/ubot+/phot) the velocity is v = ∇S, so this
#     puts a ‖∇S‖² penalty at potency-weighted cells, matching the WLF
#     action interpretation that equilibrium states should be critical
#     points of S; for RF / Helmholtz the velocity is the model output
#     directly. See compute_sink_loss for the shared implementation.
#
#   config.train.lambda_div : float           default 0.0
#     Soft divergence penalty on the curl: λ_div · mean (div(c))². Estimated
#     with a single-sample Hutchinson trace estimator
#         div(c) ≈ ε^T ∂c/∂x ε,   ε ~ N(0, I)
#     The estimator is UNBIASED for div(c) but BIASED UPWARD for div(c)²
#     (Var term adds to expectation; Jensen ⇒ over-estimate). We accept the
#     bias — qualitatively still a divergence regulariser, and tune λ_div
#     empirically rather than via theoretical magnitude matching. The
#     ``decoupled`` mode uses two independent ε samples for an unbiased
#     estimate at 2× compute; gated by ``config.train.div_estimator``.
#     REQUIRES helmholtz=True.
#
#   config.train.lambda_c : float             default 0.0
#     L2 penalty on the curl magnitude: λ_c · mean ‖c(t,x)‖². This (NOT
#     λ_div) is what makes the model converge to gradient-only as λ_c → ∞.
#     A divergence-free c can still be non-zero (e.g. constant translation),
#     so λ_div alone does NOT recover gradient-only behaviour in the limit.
#     Use λ_c as the explicit "prefer gradient-only" prior; the spec calls
#     out that for most well-curated developmental data c should be small.
#     REQUIRES helmholtz=True.
#
#   config.train.div_estimator : str          default "hutchinson"
#     "hutchinson"  — single-sample biased-on-squared (cheap, default).
#     "decoupled"   — two independent εs, unbiased (2× compute).
# ---------------------------------------------------------------------------


def _helmholtz_flag(config) -> bool:
  return bool(getattr(config.model_s, "helmholtz", False))


def _is_helmholtz(model_s) -> bool:
  """Identify a HelmholtzModelPair by class, not by config — so the helper
  works inside JIT/pmap where config may be partial."""
  return isinstance(model_s, mutils.HelmholtzModelPair)


def get_velocity_fn(model_s, params_s, train, loss_name=None):
  """Return a callable ``v(t, x, rng=None) -> (B, dim)`` giving the velocity
  at ``(t, x)``, regardless of whether ``model_s`` is:
    - a HelmholtzModelPair (.apply already returns velocity),
    - a vector-field model like ``mlp_vf`` (output IS velocity, ``rf`` loss),
    - a scalar potential like ``mlp_scalar_s`` / ``mlp_s`` (velocity = +∇_x s,
      every action-style loss in this codebase: am/sb/ubot/ubot+/phot).

  The Helmholtz pair and ``mlp_vf`` are unified: their ``.apply`` already
  returns velocity. For scalar-potential models the velocity is recovered
  via ``jax.grad`` (matches ``eval_utils.grad_vf`` convention exactly).

  ``loss_name`` is optional; when provided we can disambiguate edge cases
  without probing model output shapes inside JIT. ``loss_name='rf'``
  implies a vector-output model; anything else (and not-Helmholtz) implies
  a scalar potential.

  This is the canonical accessor — losses, sink/div helpers, and eval
  rollouts should call it rather than re-implementing the scalar vs vector
  dispatch each time.
  """
  s = mutils.get_model_fn(model_s, params_s, train=train)
  if _is_helmholtz(model_s) or loss_name == 'rf':
    return s  # output IS velocity

  # Scalar potential → +∇_x s.
  def grad_v(t, x, rng=None):
    return jax.grad(lambda _x: s(t, _x, rng).sum())(x)
  return grad_v


# ---------------------------------------------------------------------------
# Shared sink / divergence helpers — loss-agnostic.
#
# These are used by both the action-style losses (get_loss_ours) and the
# rectified-flow loss (get_loss_rf). They take a ``velocity_fn`` (from
# ``get_velocity_fn`` above) and operate on it — they don't care whether
# the underlying ``v`` came from a scalar potential's gradient, a direct
# vector-field model, or a Helmholtz pair. That's the whole reason
# get_velocity_fn exists.
# ---------------------------------------------------------------------------
def compute_sink_loss(velocity_fn, timesteps, x, weights, sink_weight, rng):
  """L_sink = sink_weight · (Σ w(p) · ‖v(x_i, t_i)‖²) / Σ w.

  Evaluates v at observed cells (per-marginal x_m, t_m) — NOT bridge samples
  — and weights by per-cell potency-derived weights w. Normalised by Σw so
  ``sink_weight`` stays dataset-invariant (a high-progenitor batch and a
  low-progenitor batch produce comparable sink magnitudes).

  Spec interpretation (action-style losses with v=∇S): "committed cells
  should be at critical points of S" — committed (low-potency) cells get
  high w(p), so the loss drives ∇S → 0 at their positions. Action stays
  flat (zero velocity) at terminal states; non-terminal action shape is
  unconstrained by this term.

  Returns ``(loss_scalar, diagnostics_dict)``."""
  n_marg = x.shape[1]

  def _v_at_marg(m_idx):
    t_m = timesteps[:, m_idx, :]                # (B, 1)
    x_m = x[:, m_idx, :]                        # (B, D)
    rng_m = random.fold_in(rng, m_idx)          # decorrelate dropout per marginal
    return velocity_fn(t_m, x_m, rng_m)         # (B, D)

  v_obs = jax.vmap(_v_at_marg)(jnp.arange(n_marg))   # (n_marg, B, D)
  norms_sq = (v_obs ** 2).sum(-1)                     # (n_marg, B)
  # OT-coupling code path threads weights alongside cells through an iterator
  # that adds a trailing singleton dim (treats per-cell-weight as if it were
  # a feature vector of size 1), so `weights` arrives as (B, n_marg, 1)
  # instead of the linear-coupling (B, n_marg). Squeeze the spurious axis so
  # this function is shape-robust to both iterators.
  weights = jnp.asarray(weights)
  if weights.ndim == 3 and weights.shape[-1] == 1:
    weights = weights.squeeze(-1)
  w = weights.transpose((1, 0))                       # (n_marg, B)
  w_sum = w.sum() + 1e-8
  loss_sink = sink_weight * (w * norms_sq).sum() / w_sum
  diag = {
    'loss_sink': loss_sink,
    'v_rms_weighted': jnp.sqrt(
      (w * norms_sq).sum() / w_sum + 1e-12),
  }
  return loss_sink, diag


def compute_div_v_hutchinson(velocity_fn, t, x, rng):
  """Single-sample Hutchinson estimate of div(v) = trace(∂v/∂x), evaluated
  at ``(t, x)``. Returns the per-sample estimate ``(B,)`` — call ``.mean()``
  for the population mean, ``(.**2).mean()`` for the squared mean.

  For v = ∇S (action losses) this estimates ΔS (Laplacian of the action);
  for Helmholtz v = ∇U + c it estimates div(v) = ΔU + div(c). The full-
  velocity divergence is the right monitor regardless of parameterisation
  — it tells you "is the model contracting / expanding mass at this
  point", which is the local mass-conservation residual under continuity.
  Useful for spotting attractor cores (very negative div(v) = strong
  contraction) and source-like regions (positive)."""
  eps = random.normal(rng, x.shape)
  _, jvp = jax.jvp(lambda _x: velocity_fn(t, _x, rng), (x,), (eps,))
  return (jvp * eps).sum(-1)  # (B,) — unbiased estimate of div(v)


def get_loss(config, model_s, model_q, time_sampler, train):
  if _helmholtz_flag(config) and config.loss not in ('rf', 'hybrid_am_helm'):
    raise NotImplementedError(
      f"config.model_s.helmholtz=True is supported with config.loss in "
      f"('rf', 'hybrid_am_helm'). Got config.loss={config.loss!r}. AM-"
      f"family losses (am/sb/ubot/ubot+/phot) require v=∇S for their HJB "
      f"derivation; adding a non-gradient curl breaks the identification. "
      f"To use Helmholtz on a non-RF host, use 'hybrid_am_helm': it trains "
      f"U via AM's action term and c via an RF-style chord-matching term, "
      f"composing the two with `train.lambda_match`."
    )
  if config.loss == 'am':
    return get_loss_ours(config, model_s, model_q, time_sampler, train)
  elif config.loss == 'phot':
    return get_loss_ours(config, model_s, model_q, time_sampler, train)
  elif config.loss == 'sb':
    return get_loss_ours(config, model_s, model_q, time_sampler, train)
  elif config.loss == 'ubot':
    return get_loss_ours(config, model_s, model_q, time_sampler, train)
  elif config.loss == 'ubot+':
    return get_loss_ours(config, model_s, model_q, time_sampler, train)
  elif config.loss == 'rf':
    return get_loss_rf(config, model_s, model_q, time_sampler, train)
  elif config.loss == 'hybrid_am_helm':
    return get_loss_hybrid_am_helmholtz(
      config, model_s, model_q, time_sampler, train,
    )
  else:
    NotImplementedError(f'config.loss: {config.loss} is not implemented')


def get_loss_ours(config, model_s, model_q, time_sampler, train):
  
  if config.loss == 'am':
    def potential(_t, _x, _key, _s):
      dsdtdx_fn = jax.grad(lambda __t, __x, __key: _s(__t, __x, __key).sum(), argnums=[0,1])
      dsdt, dsdx = dsdtdx_fn(_t, _x, _key)
      return dsdt + 0.5*(dsdx**2).sum(1, keepdims=True)
  elif config.loss == 'phot':
    physical_potential = get_toy_physical_potential(config)
    def potential(_t, _x, _key, _s):
      dsdtdx_fn = jax.grad(lambda __t, __x, __key: _s(__t, __x, __key).sum(), argnums=[0,1])
      dsdt, dsdx = dsdtdx_fn(_t, _x, _key)
      return dsdt + 0.5*(dsdx**2).sum(1, keepdims=True) + physical_potential(_t, _x)
  elif config.loss == 'sb':
    def potential(_t, _x, _key, _s):
      keys = random.split(_key, 2)
      dsdt_fn = jax.grad(lambda __t, __x, __key: _s(__t, __x, __key).sum(), argnums=0)
      dsdx_fn = jax.grad(lambda __t, __x, __key: _s(__t, __x, __key).sum(), argnums=1)
      
      eps = random.randint(keys[0], _x.shape, 0, 2).astype(float)*2 - 1.0
      dsdx_val, jvp_val = jax.jvp(lambda __x: dsdx_fn(_t, __x, keys[1]), (_x,), (eps,))
      dsdt_val = dsdt_fn(_t, _x, keys[1])
      out = dsdt_val + 0.5*(dsdx_val**2).sum(1, keepdims=True)
      out += 0.5*config.sigma**2*(jvp_val*eps).sum(1, keepdims=True)
      return out
  elif config.loss == 'ubot':
    def potential(_t, _x, _key, _s):
      dsdtdx_fn = jax.grad(lambda __t, __x, __key: _s(__t, __x, __key).sum(), argnums=[0,1])
      dsdt, dsdx = dsdtdx_fn(_t, _x, _key)
      return dsdt + 0.5*(dsdx**2).sum(1, keepdims=True) + config.lambd*0.5*(_s(_t, _x, _key))**2
  elif config.loss == 'ubot+':
    physical_potential = get_physical_potential(config)
    def potential(_t, _x, _key, _s):
      dsdtdx_fn = jax.grad(lambda __t, __x, __key: _s(__t, __x, __key).sum(), argnums=[0,1])
      dsdt, dsdx = dsdtdx_fn(_t, _x, _key)
      return dsdt + 0.5*(dsdx**2).sum(1, keepdims=True) + config.lambd*0.5*(_s(_t, _x, _key)**2) + physical_potential(_t, _x)
  else:
    NotImplementedError(f'potential for config.loss: {config.loss} is not implemented')

  # ---- lfvae extension knobs (sink + div diagnostic) ----
  # These work for every action-style loss (am/sb/ubot/ubot+/phot) because
  # they all share the v = ∇S velocity identification — see
  # ``get_velocity_fn`` for the dispatch. Helmholtz (curl + lambda_c +
  # lambda_div as a loss term) is NOT supported here — see get_loss().
  sink_weight = float(getattr(config.train, "sink_weight", 0.0))
  emit_div_diag = True  # cheap; always emit so sweeps can compare runs

  def loss_fn(key, params_s, params_q, sampler_state, batch):
    # Optional sink weights at the tail of the batch tuple (3-tuple when
    # sink supervision is active; 2-tuple otherwise — see datasets.py).
    if len(batch) == 3:
      timesteps, x, weights = batch
    else:
      timesteps, x = batch
      weights = None
    bs = x.shape[0]

    keys = random.split(key, num=10)
    s = mutils.get_model_fn(model_s, params_s, train=train)
    q = mutils.get_model_fn(model_q, params_q, train=train)

    ################################################# loss s #################################################
    acceleration_fn = jax.grad(lambda _t, _x, _key: potential(_t, _x, _key, s).sum(), argnums=1)

    # sample time
    t_0, t_1 = timesteps[:,0,:], timesteps[:,-1,:]
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    t = t.reshape(-1,1)

    # The bridge q model (mlp_q) always expects the 2-tuple (timesteps, x).
    # When sink supervision is active, ``batch`` is the 3-tuple (t, x, w) —
    # pass the 2-tuple form explicitly so q's ``timesteps, x = batch``
    # unpack stays compatible whether or not sink is on.
    batch_for_q = (timesteps, x)

    # sample data
    samples_q = q(t, batch_for_q, keys[0])
    x_t = jax.lax.stop_gradient(samples_q)
    mask = (t >= timesteps[:,:-1,0])*(t <= timesteps[:,1:,0])
    t_mult = config.train.step_size*((1.0 - ((t-timesteps[:,:-1,0])/(timesteps[:,1:,0]-timesteps[:,:-1,0]))**2*mask -\
        ((timesteps[:,1:,0]-t)/(timesteps[:,1:,0]-timesteps[:,:-1,0]))**2*mask)*mask).sum(1, keepdims=True)
    for i in range(config.train.n_gradient_steps):
      dx = jax.lax.stop_gradient(acceleration_fn(t, x_t, jax.random.fold_in(keys[1], i)))
      x_t = x_t + t_mult*jnp.clip(dx, -1, 1)
    
    # boundaries loss
    x_0, x_1 = x[:,0,:], x[:,-1,:]
    s_0 = s(t_0, x_0, keys[2])
    s_1 = s(t_1, x_1, keys[3])
    loss_s = s_0.reshape((-1,1)) - s_1.reshape((-1,1))
    print(loss_s.shape, 'boundaries.shape', flush=True)

    # time loss
    potential_value = potential(t, x_t, keys[4], s)
    loss_s += potential_value
    print(loss_s.shape, 'final.shape', flush=True)
    metrics = {}
    metrics['loss_s'] = loss_s.mean()
    total_loss = loss_s.mean()
    
    ################################################# loss q #################################################

    s_detached = mutils.get_model_fn(model_s, jax.lax.stop_gradient(params_s), train=train)
    loss_q = -potential(t, samples_q, keys[5], s_detached)
    metrics['loss_q'] = loss_q.mean()
    total_loss += loss_q.mean()

    metrics['acceleration'] = jnp.linalg.norm(acceleration_fn(t, samples_q, keys[6]), axis=1).mean()
    potential_value = jax.lax.stop_gradient(potential_value.squeeze())
    metrics['potential_var'] = ((potential_value.mean() - potential_value)**2).mean()

    # ------------------------------------------------------------------
    # lfvae extensions: sink supervision (loss term, gated) and
    # divergence-of-velocity diagnostic (monitor only, always on).
    #
    # SINK (loss term, sink_weight > 0): with v = ∇S for action losses,
    # this puts a per-cell penalty ‖∇S(x_i, t_i)‖² weighted by potency
    # — committed cells should be at critical points of S, which is what
    # the WLF action interpretation expects of equilibrium states.
    #
    # DIVERGENCE (monitor only): div(v) = ΔS for v = ∇S. Useful as a
    # sanity check — strongly negative ΔS at observed cells indicates
    # attractor cores, positive indicates sources. Always emitted; the
    # cost is one Hutchinson-trace evaluation per step (cheap).
    # ------------------------------------------------------------------
    velocity_fn = get_velocity_fn(model_s, params_s, train=train,
                                   loss_name=config.loss)
    if sink_weight > 0 and weights is not None:
      loss_sink, sink_diag = compute_sink_loss(
        velocity_fn, timesteps, x, weights, sink_weight, keys[7])
      total_loss = total_loss + loss_sink
      metrics.update(sink_diag)
    if emit_div_diag:
      div_v = compute_div_v_hutchinson(velocity_fn, t, x_t, keys[8])
      metrics['div_v_rms'] = jnp.sqrt(jnp.clip((div_v ** 2).mean(), 1e-12))
      metrics['div_v_mean'] = div_v.mean()  # signed; spots net source/sink

    return total_loss, (next_sampler_state, metrics)

  return loss_fn

def get_loss_rf(config, model_s, model_q, time_sampler, train):
  """Rectified-flow / flow-matching loss with optional lfvae extensions.

  Base term (always on): MSE between the model's velocity and the linear
  interpolant velocity between consecutive observed marginals (Liu et al.
  2022, Lipman et al. 2023). Identical to upstream when no extensions are
  enabled — recover by setting helmholtz=False, sink_weight=lambda_div=
  lambda_c=0 and using a vector-output ``model_s`` (e.g. ``mlp_vf``).

  Optional extensions, gated by config flags (see module docstring at top):
    - Helmholtz velocity v = ∇U + c (config.model_s.helmholtz)
    - Sink supervision    L_sink = (Σ w·‖v‖²)/Σ w (config.train.sink_weight)
    - Divergence reg      L_div  = λ_div · (div(c))² (config.train.lambda_div)
    - Curl-magnitude L2   L_c    = λ_c · ‖c‖²        (config.train.lambda_c)
  """
  helmholtz = _is_helmholtz_config(config)
  sink_weight = float(getattr(config.train, "sink_weight", 0.0))
  lambda_div = float(getattr(config.train, "lambda_div", 0.0))
  lambda_c = float(getattr(config.train, "lambda_c", 0.0))
  div_estimator = str(getattr(config.train, "div_estimator", "hutchinson"))

  if (lambda_div > 0 or lambda_c > 0) and not helmholtz:
    raise ValueError(
      "config.train.lambda_div and lambda_c regularize the curl head, "
      "which only exists when config.model_s.helmholtz=True. Got "
      f"helmholtz=False, lambda_div={lambda_div}, lambda_c={lambda_c}."
    )

  def loss_fn(key, params_s, params_q, sampler_state, batch):
    # Batch tuple shape:
    #   (timesteps, x)         — no sink supervision (default)
    #   (timesteps, x, w)      — sink supervision active (per-cell potency)
    # The third element is plumbed through datasets.linear_train_iterator
    # when config.train.sink_weight > 0 (see datasets.py).
    if len(batch) == 3:
      timesteps, x, weights = batch
    else:
      timesteps, x = batch
      weights = None
    bs = x.shape[0]

    keys = random.split(key, num=12)
    s = mutils.get_model_fn(model_s, params_s, train=train)
    q = mutils.get_model_fn(model_q, params_q, train=train)

    # ----------------------------------------------------------------------
    # Flow-matching base loss (loss_s in upstream).
    # ----------------------------------------------------------------------
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    t = t.reshape(-1, 1)

    # timesteps shape: (batch, n_marginals, 1); x shape: (batch, n_marginals, dim).
    t_right = (timesteps < t.reshape(-1, 1, 1)).sum(1)
    t_right = jnp.fmax(t_right, jnp.ones_like(t_right).astype(int))
    t_left = t_right - 1

    x_left = x[jnp.arange(len(x)), t_left.ravel(), :]
    x_right = x[jnp.arange(len(x)), t_right.ravel(), :]
    t_0 = timesteps[jnp.arange(len(x)), t_left.ravel(), :]
    t_1 = timesteps[jnp.arange(len(x)), t_right.ravel(), :]

    x_t = (t_1 - t) / (t_1 - t_0) * x_left + (t - t_0) / (t_1 - t_0) * x_right
    dxtdt = (x_right - x_left) / (t_1 - t_0)

    v = s(t, x_t, keys[1])
    loss_fm = ((v - dxtdt) ** 2).sum(-1, keepdims=True)
    metrics = {'loss_fm': loss_fm.mean()}
    total_loss = loss_fm.mean()

    # ----------------------------------------------------------------------
    # Sink supervision (delegated to the shared compute_sink_loss helper —
    # same math used by action-style losses in get_loss_ours).
    # ----------------------------------------------------------------------
    velocity_fn_obs = get_velocity_fn(model_s, params_s, train=train,
                                       loss_name='rf')
    if sink_weight > 0 and weights is not None:
      loss_sink, sink_diag = compute_sink_loss(
        velocity_fn_obs, timesteps, x, weights, sink_weight, keys[2])
      total_loss = total_loss + loss_sink
      metrics.update(sink_diag)

    # ----------------------------------------------------------------------
    # Divergence-of-velocity diagnostic (monitor only — full v, not just c).
    # Same helper as get_loss_ours so the metric is comparable across all
    # losses in the sweep. For Helmholtz this is div(∇U + c) = ΔU + div(c);
    # for non-Helmholtz RF it's div(v_θ).
    # ----------------------------------------------------------------------
    div_v = compute_div_v_hutchinson(velocity_fn_obs, t, x_t, keys[7])
    metrics['div_v_rms'] = jnp.sqrt(jnp.clip((div_v ** 2).mean(), 1e-12))
    metrics['div_v_mean'] = div_v.mean()

    # ----------------------------------------------------------------------
    # Divergence regularisation: Hutchinson estimate of (div c)² at x_t.
    # Single-sample form is BIASED upward on the squared quantity (Var term
    # adds to the expectation); we accept this — see module docstring. Use
    # "decoupled" estimator for an unbiased 2-sample variant at 2× compute.
    # ----------------------------------------------------------------------
    if lambda_div > 0:
      def _c_call(t_, x_, rng_):
        # apply_curl uses the same rng pathway as get_model_fn so dropout
        # behaves consistently with the rest of the loss when train=True.
        rngs = {"dropout": rng_} if train else None
        return model_s.apply_curl({"params": params_s}, t_, x_,
                                   train=train, mutable=False, rngs=rngs)

      def _div_hutch(rng_):
        eps = random.normal(rng_, x_t.shape)
        _, jvp = jax.jvp(lambda _x: _c_call(t, _x, rng_), (x_t,), (eps,))
        return (jvp * eps).sum(-1)  # (B,) — unbiased estimate of div(c)

      if div_estimator == "decoupled":
        d1 = _div_hutch(keys[3])
        d2 = _div_hutch(keys[4])
        div_sq = d1 * d2          # unbiased estimate of (div c)²
      else:
        div_one = _div_hutch(keys[3])
        div_sq = div_one ** 2     # biased upward; documented

      loss_div = lambda_div * div_sq.mean()
      total_loss = total_loss + loss_div
      metrics['loss_div'] = loss_div
      metrics['div_c_rms'] = jnp.sqrt(jnp.clip(div_sq.mean(), 1e-12))

    # ----------------------------------------------------------------------
    # Curl-magnitude L2: λ_c · mean ‖c(t, x_t)‖²
    # Together with λ_div=0 this is the lever for "prefer gradient-only" — at
    # high λ_c the curl head shrinks to zero and the Helmholtz model
    # collapses to the pure-gradient baseline. λ_div ALONE does not produce
    # this collapse (a divergence-free c can still be large).
    # ----------------------------------------------------------------------
    if lambda_c > 0:
      rngs_c = {"dropout": keys[5]} if train else None
      c_at_xt = model_s.apply_curl({"params": params_s}, t, x_t,
                                    train=train, mutable=False, rngs=rngs_c)
      norm_c_sq = (c_at_xt ** 2).sum(-1)  # (B,)
      loss_c_norm = lambda_c * norm_c_sq.mean()
      total_loss = total_loss + loss_c_norm
      metrics['loss_c_norm'] = loss_c_norm
      metrics['c_norm_rms'] = jnp.sqrt(jnp.clip(norm_c_sq.mean(), 1e-12))

    # ----------------------------------------------------------------------
    # Bridge q acceleration diagnostic (unchanged from upstream).
    # ----------------------------------------------------------------------
    dvdt = jax.jacrev(lambda _t: s(_t, x_t, keys[6]).sum(0))(t)
    dvdt = jnp.squeeze(dvdt).transpose((1, 0))
    dvdxv = jax.jvp(lambda _x: s(t, _x, keys[6]), (x_t,), (v,))[0]
    acceleration = dvdt + dvdxv
    metrics['acceleration'] = jnp.linalg.norm(acceleration, axis=1).mean()

    return total_loss, (next_sampler_state, metrics)

  return loss_fn


def _is_helmholtz_config(config) -> bool:
  return bool(getattr(config.model_s, "helmholtz", False))


# ---------------------------------------------------------------------------
# Hybrid AM-Helmholtz loss
#
# The motivation, recapped (per the implementation-plan discussion):
#
#   - WLF's action-matching losses (am/sb/ubot/ubot+/phot) bake in
#     v = ∇S — the HJB derivation identifies the velocity with the
#     gradient of the scalar action. Substituting v = ∇U + c into the
#     kinetic term destroys the identification: ∂L/∂c = E[v] = 0, which
#     drives the average velocity to zero (silently wrong).
#
#   - Rectified flow (RF) doesn't have this issue (the loss is just
#     ``mean(‖v − target‖²)`` with no HJB structure), so 'rf'+Helmholtz is
#     mathematically clean and was the original Helmholtz host.
#
#   - On Walchli, RF turned out to be a much weaker baseline than AM
#     (init_final_corr 0.36–0.50 vs 0.92–0.95 for am_otcouple). So
#     Helmholtz-on-RF can't be cleanly evaluated: the host model is broken.
#
# The Hybrid AM-Helmholtz approach gets around this by COMPOSING two
# losses with different targets:
#
#   1. AM's HJB loss on U alone — preserves the action structure on the
#      scalar potential. U trains as it would under pure AM.
#   2. An RF-style chord-matching term on v = ∇U + c — supervises c via
#      the same flow-matching objective Helmholtz was clean for on RF.
#
#       L = L_AM(U) + λ_match · L_FM(v = ∇U + c)
#
# Interpretation: U is the action (cells flow downhill); c is the residual
# vector field needed to match the marginal transport that U alone can't.
# If U is sufficient (the data's transport is gradient-only), λ_c at any
# positive value drives c to zero and the model behaves as pure AM. If U
# is insufficient (barycentric-field failure), c picks up the residual.
#
# Online references:
#   * Neklyudov et al. ICML 2024 (WLF) for the AM derivation.
#   * Lipman et al. 2023 (arXiv:2210.02747) for the parameterisation-
#     agnostic flow-matching loss used as L_FM.
#   * Tong et al. 2023 (OT-CFM, arXiv:2302.00482) for the consecutive-pair
#     bin-couple chord-matching pattern, which we mirror for L_FM.
# ---------------------------------------------------------------------------
def get_potential_fn(model_s, params_s, train):
  """Return a callable ``U(t, x, rng=None) -> (B, 1)`` for the SCALAR
  potential head.

  Symmetric counterpart to :func:`get_velocity_fn`:
    - For a HelmholtzModelPair, calls ``.apply_potential`` which returns
      U (scalar) without invoking the curl head.
    - For a non-Helmholtz scalar model, the model itself IS the potential
      (its ``.apply`` returns scalar) — so we use ``get_model_fn``
      directly.

  The hybrid AM-Helmholtz loss uses this for the AM/HJB term: AM's
  derivation only needs U, not v, so we route the AM math through the
  scalar potential head exclusively.
  """
  if _is_helmholtz(model_s):
    # HelmholtzModelPair-specific accessor that wraps .apply_potential the
    # same way mutils.get_model_fn wraps .apply.
    def u_fn(t, x, rng=None):
      variables = dict(params=params_s)
      if not train:
        return model_s.apply_potential(
          variables, t, x, train=False, mutable=False,
        )
      rngs = {'dropout': rng}
      return model_s.apply_potential(
        variables, t, x, train=True, mutable=False, rngs=rngs,
      )
    return u_fn
  # Non-Helmholtz: ``model_s`` IS the scalar potential.
  return mutils.get_model_fn(model_s, params_s, train=train)


def get_loss_hybrid_am_helmholtz(
  config, model_s, model_q, time_sampler, train,
):
  """Hybrid AM-Helmholtz loss factory: AM on U + λ_match · FM on (∇U + c).

  Requires ``config.model_s.helmholtz = True`` (the curl head needs to
  exist to be trained). Adds three knobs on top of the WLF AM defaults:

  * ``config.train.lambda_match`` (float, default 1.0) — relative weight of
    the FM chord-matching term that supervises ``v = ∇U + c``. At
    λ_match=0 the curl head receives no gradient signal and decays to zero
    under ``λ_c > 0``, recovering pure AM. At λ_match=1.0 the two terms
    contribute comparably; higher values prioritise marginal matching
    (the c head gets stronger updates relative to the action structure on U).

  * ``config.train.sink_weight`` — same potency-sink term as the other
    losses; evaluated on observed cells using the full Helmholtz velocity.

  * ``config.train.lambda_div`` / ``lambda_c`` — curl-head regularizers.

  Structurally mirrors :func:`get_loss_ours` for the AM part (the
  ``potential`` closure → bridge q → boundary + time-integral) and
  :func:`get_loss_rf` for the FM part (consecutive-bin chord matching).
  """
  if not _is_helmholtz_config(config):
    raise ValueError(
      "loss='hybrid_am_helm' requires config.model_s.helmholtz=True (the "
      "curl head must exist to be trained)."
    )

  lambda_match = float(getattr(config.train, "lambda_match", 1.0))
  sink_weight = float(getattr(config.train, "sink_weight", 0.0))
  lambda_div = float(getattr(config.train, "lambda_div", 0.0))
  lambda_c = float(getattr(config.train, "lambda_c", 0.0))
  div_estimator = str(getattr(config.train, "div_estimator", "hutchinson"))

  # AM term operates on U (scalar potential). Its ``potential`` closure is
  # the standard AM Lagrangian density: dU/dt + 0.5·‖∇U‖² (the kinetic
  # half-energy on the gradient flow ∇U, NOT on the full Helmholtz v).
  # Critically: c is NOT in this closure. AM trains U as if it were a
  # gradient-only model. The c head is supervised separately by L_FM.
  def potential_U(_t, _x, _key, _u):
    dudtdx_fn = jax.grad(
      lambda __t, __x, __key: _u(__t, __x, __key).sum(), argnums=[0, 1],
    )
    dudt, dudx = dudtdx_fn(_t, _x, _key)
    return dudt + 0.5 * (dudx ** 2).sum(1, keepdims=True)

  def loss_fn(key, params_s, params_q, sampler_state, batch):
    if len(batch) == 3:
      timesteps, x, weights = batch
    else:
      timesteps, x = batch
      weights = None
    bs = x.shape[0]

    keys = random.split(key, num=14)
    u_fn = get_potential_fn(model_s, params_s, train=train)
    v_fn = mutils.get_model_fn(model_s, params_s, train=train)  # Helmholtz pair → velocity
    q = mutils.get_model_fn(model_q, params_q, train=train)

    # ----- AM term on U (mirrors get_loss_ours['am']) -----
    acceleration_fn = jax.grad(
      lambda _t, _x, _key: potential_U(_t, _x, _key, u_fn).sum(), argnums=1,
    )
    t_0, t_1 = timesteps[:, 0, :], timesteps[:, -1, :]
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    t = t.reshape(-1, 1)

    # Bridge q samples + AM inner-loop refinement (verbatim from
    # get_loss_ours so the AM term's training dynamics match upstream).
    # Re-pack the 2-tuple form mlp_q expects (it does ``timesteps, x = batch``),
    # since ``batch`` may be a 3-tuple when sink supervision is active.
    batch_for_q = (timesteps, x)
    samples_q = q(t, batch_for_q, keys[0])
    x_t = jax.lax.stop_gradient(samples_q)
    mask = (t >= timesteps[:, :-1, 0]) * (t <= timesteps[:, 1:, 0])
    t_mult = config.train.step_size * (
      (1.0 - ((t - timesteps[:, :-1, 0]) /
              (timesteps[:, 1:, 0] - timesteps[:, :-1, 0])) ** 2 * mask
       - ((timesteps[:, 1:, 0] - t) /
          (timesteps[:, 1:, 0] - timesteps[:, :-1, 0])) ** 2 * mask) * mask
    ).sum(1, keepdims=True)
    for i in range(config.train.n_gradient_steps):
      dx = jax.lax.stop_gradient(
        acceleration_fn(t, x_t, jax.random.fold_in(keys[1], i)),
      )
      x_t = x_t + t_mult * jnp.clip(dx, -1, 1)

    # AM boundary + time-integral.
    x0_b, x1_b = x[:, 0, :], x[:, -1, :]
    u_0 = u_fn(t_0, x0_b, keys[2])
    u_1 = u_fn(t_1, x1_b, keys[3])
    loss_AM = u_0.reshape((-1, 1)) - u_1.reshape((-1, 1))
    pot_value = potential_U(t, x_t, keys[4], u_fn)
    loss_AM = loss_AM + pot_value
    metrics = {'loss_AM': loss_AM.mean()}
    total_loss = loss_AM.mean()

    # ----- FM term on v = ∇U + c (mirrors get_loss_rf chord matching) -----
    # Use the SAME sampled t and reuse x_t? NO — RF needs a clean linear
    # interpolant between consecutive bins, not the bridge-refined point.
    # Sample a new (consecutive bin pair, t, x_fm) using RF's interpolant
    # logic.
    t_right = (timesteps < t.reshape(-1, 1, 1)).sum(1)
    t_right = jnp.fmax(t_right, jnp.ones_like(t_right).astype(int))
    t_left = t_right - 1
    x_left = x[jnp.arange(len(x)), t_left.ravel(), :]
    x_right = x[jnp.arange(len(x)), t_right.ravel(), :]
    t_0_pair = timesteps[jnp.arange(len(x)), t_left.ravel(), :]
    t_1_pair = timesteps[jnp.arange(len(x)), t_right.ravel(), :]
    x_t_fm = (
      (t_1_pair - t) / (t_1_pair - t_0_pair) * x_left
      + (t - t_0_pair) / (t_1_pair - t_0_pair) * x_right
    )
    dxtdt = (x_right - x_left) / (t_1_pair - t_0_pair)

    v_at_xt = v_fn(t, x_t_fm, keys[5])   # Helmholtz pair → v = ∇U + c
    loss_FM = ((v_at_xt - dxtdt) ** 2).sum(-1, keepdims=True).mean()
    metrics['loss_FM'] = loss_FM
    total_loss = total_loss + lambda_match * loss_FM

    # ----- Sink supervision (loss-agnostic helper) -----
    if sink_weight > 0 and weights is not None:
      loss_sink, sink_diag = compute_sink_loss(
        v_fn, timesteps, x, weights, sink_weight, keys[6],
      )
      total_loss = total_loss + loss_sink
      metrics.update(sink_diag)

    # ----- Divergence regulariser on c (Hutchinson) -----
    if lambda_div > 0:
      def _c_call(t_, x_, rng_):
        rngs = {"dropout": rng_} if train else None
        return model_s.apply_curl(
          {"params": params_s}, t_, x_,
          train=train, mutable=False, rngs=rngs,
        )

      def _div_hutch(rng_):
        eps = random.normal(rng_, x_t_fm.shape)
        _, jvp = jax.jvp(lambda _x: _c_call(t, _x, rng_), (x_t_fm,), (eps,))
        return (jvp * eps).sum(-1)

      if div_estimator == "decoupled":
        d1 = _div_hutch(keys[7])
        d2 = _div_hutch(keys[8])
        div_sq = d1 * d2
      else:
        d_one = _div_hutch(keys[7])
        div_sq = d_one ** 2

      loss_div = lambda_div * div_sq.mean()
      total_loss = total_loss + loss_div
      metrics['loss_div'] = loss_div
      metrics['div_c_rms'] = jnp.sqrt(jnp.clip(div_sq.mean(), 1e-12))

    # ----- Curl-magnitude L2 -----
    if lambda_c > 0:
      rngs_c = {"dropout": keys[9]} if train else None
      c_at_xt = model_s.apply_curl(
        {"params": params_s}, t, x_t_fm,
        train=train, mutable=False, rngs=rngs_c,
      )
      norm_c_sq = (c_at_xt ** 2).sum(-1)
      loss_c_norm = lambda_c * norm_c_sq.mean()
      total_loss = total_loss + loss_c_norm
      metrics['loss_c_norm'] = loss_c_norm
      metrics['c_norm_rms'] = jnp.sqrt(jnp.clip(norm_c_sq.mean(), 1e-12))

    # ----- Bridge q's adversarial loss (verbatim from AM) -----
    # The bridge is trained against U only, NOT v=∇U+c — keeping the q
    # objective aligned with the action structure that defined it.
    u_detached = get_potential_fn(
      model_s, jax.lax.stop_gradient(params_s), train=train,
    )
    loss_q = -potential_U(t, samples_q, keys[10], u_detached)
    metrics['loss_q'] = loss_q.mean()
    total_loss = total_loss + loss_q.mean()

    # Acceleration diagnostic (matches AM's) — measures the gradient of
    # the AM potential closure on the bridge samples.
    metrics['acceleration'] = jnp.linalg.norm(
      acceleration_fn(t, samples_q, keys[11]), axis=1,
    ).mean()
    return total_loss, (next_sampler_state, metrics)

  return loss_fn

import datasets
import numpy as np



def get_physical_potential(config):
  init_key = random.PRNGKey(0)
  X, _, _, _, _ = datasets.get_data(config, init_key)
  t = np.linspace(0.0, 1.0, len(X)).tolist()
  
  t_grid = []
  acc_grid = []
  for i in range(1, len(X)-1):
    v_prev = (X[i].mean(0) - X[i-1].mean(0))/(t[i]-t[i-1])
    v_next = (X[i+1].mean(0) - X[i].mean(0))/(t[i+1]-t[i])
    acc_grid.append((v_next - v_prev)/(0.5*(t[i+1]+t[i]) - 0.5*(t[i]-t[i-1])))
    t_grid.append(t[i])
  t_grid = jnp.array(t_grid)
  acc_grid = jnp.stack(acc_grid)
  max_acc = jnp.max(jnp.linalg.norm(acc_grid, axis=1))
  
  def potential(t, x):
    ids = jnp.argmin(jnp.abs(t - t_grid[None,:]), axis=1)
    out = -(x*acc_grid[ids]).sum(1, keepdims=True)
    out = jnp.clip(out, -1e4, max_acc)
    return out
    
  return potential

# def get_physical_potential(config):
#   init_key = random.PRNGKey(0)
#   X, _ = datasets.get_data(config, init_key)
#   t = np.linspace(0.0, 1.0, len(X)).tolist()
  
#   t_grid = []
#   acc_grid = []
#   for i in range(1, len(X)-1):
#     v_prev = (X[i].mean(0) - X[i-1].mean(0))/(t[i]-t[i-1])
#     v_next = (X[i+1].mean(0) - X[i].mean(0))/(t[i+1]-t[i])
#     acc_grid.append((v_next - v_prev)/(0.5*(t[i+1]+t[i]) - 0.5*(t[i]-t[i-1])))
#     t_grid.append(t[i])
#   t_grid = jnp.array(t_grid)
#   acc_grid = jnp.stack(acc_grid)
#   max_acc = jnp.max(jnp.linalg.norm(acc_grid, axis=1))
  
#   def potential(t, x):
#     ids = jnp.argmin(jnp.abs(t - t_grid[None,:]), axis=1)
#     out = -(x*acc_grid[ids]).sum(1, keepdims=True)
#     out = jnp.clip(out, -1e4, max_acc)
#     return out
    
#   return potential

def get_toy_physical_potential(config):
  def potential(t, x):
    out = (x**2).sum(1, keepdims=True)
    return out
    
  return potential
