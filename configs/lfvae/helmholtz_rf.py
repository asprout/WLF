"""Reference config for Helmholtz-decomposed velocity with rectified-flow
loss (lfvae extension to wl-mechanics).

This is a **documentation** artifact, not the primary entry point. The
lfvae pipeline builds its config programmatically inside
``tools/wlf_import_worker.py:build_config`` so it can pull defaults from
``src/dynamics/defaults.py:WLFConfig`` and override per-CLI flag. This
file is here for anyone who wants to run ``main.py --config ...`` directly
against the wl-mechanics CLI, e.g. to debug a Helmholtz training on a toy
problem outside the lfvae pipeline.

Knob meanings (full justification in
``envs/dynamics-wlf/wl-mechanics/losses.py`` module docstring):

  model_s.helmholtz       : parameterise v = +∇U + c instead of v = +∇S
  model_s.curl_name       : flax-registered name for the curl head
  train.sink_weight       : per-cell ‖v‖² loss weighted by potency
  train.lambda_div        : Hutchinson-estimated (div c)² regulariser
  train.div_estimator     : "hutchinson" (single ε, biased on squared) or
                            "decoupled" (two εs, unbiased, 2× compute)
  train.lambda_c          : ‖c‖² regulariser — the lever for "prefer
                            gradient-only" (λ_div alone does NOT collapse
                            to gradient-only since divergence-free c can
                            still be non-zero)
"""
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    # Helmholtz currently requires loss='rf' — see losses.get_loss for the
    # NotImplementedError raised on (helmholtz=True, loss in {am,sb,...}).
    config.loss = 'rf'
    # Linear interpolant: the OT-coupling path shuffles cell identities and
    # cannot preserve per-cell sink-weight alignment (see datasets.py).
    config.interpolant = 'linear'

    # data — placeholder values for toy / docs use; the lfvae pipeline
    # overrides these via the worker.
    config.data = data = ml_collections.ConfigDict()
    data.task = 'OT'
    data.name = 'cite'
    data.dim = 8
    data.whiten = False
    data.test_id = None
    data.t_0, data.t_1 = 0.0, 1.0

    # ---- model_s: Helmholtz potential + curl ----
    config.model_s = model_s = ml_collections.ConfigDict()
    model_s.input_dim = data.dim
    model_s.name = 'mlp_scalar_s'   # potential head (scalar)
    model_s.curl_name = 'mlp_curl'  # curl head (zero-init vector field)
    model_s.helmholtz = True
    model_s.ema_rate = 0.999
    model_s.nonlinearity = 'swish'
    model_s.nf = 256
    model_s.n_layers = 2
    model_s.skip = False
    model_s.embed_time = True
    model_s.dropout = 0.0

    # model_q: unchanged bridge (lfvae extensions don't touch q).
    config.model_q = model_q = ml_collections.ConfigDict()
    model_q.input_dim = data.dim
    model_q.n_marginals = 3
    model_q.name = 'mlp_q'
    model_q.ema_rate = 0.999
    model_q.nonlinearity = 'swish'
    model_q.nf = 256
    model_q.n_layers = 0
    model_q.skip = False
    model_q.indicator = True
    model_q.dropout = 0.0

    # opts — upstream rf defaults (1e-3, AdamW).
    config.optimizer_s = optimizer_s = ml_collections.ConfigDict()
    optimizer_s.name = 'adamw'
    optimizer_s.lr = 1e-3
    optimizer_s.beta1 = 0.9
    optimizer_s.eps = 1e-8
    optimizer_s.warmup = 5_000
    optimizer_s.grad_clip = 1.0

    config.optimizer_q = optimizer_q = ml_collections.ConfigDict()
    optimizer_q.name = 'adamw'
    optimizer_q.lr = 1e-3
    optimizer_q.beta1 = 0.9
    optimizer_q.eps = 1e-8
    optimizer_q.warmup = 5_000
    optimizer_q.grad_clip = 1.0

    # training — base RF loop + lfvae extensions (defaults match WLFConfig).
    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 512
    train.n_gradient_steps = 10
    train.step_size = 1e-2
    train.n_jitted_steps = 1
    train.n_iters = 50_000
    train.save_every = 50_000
    train.eval_every = 12_500
    train.log_every = 2_500
    # ---- lfvae extensions ----
    train.sink_weight = 0.0           # > 0 requires per-cell potency weights
    train.lambda_div = 0.0            # > 0 requires helmholtz=True
    train.div_estimator = 'hutchinson'
    train.lambda_c = 0.0              # > 0 requires helmholtz=True

    config.metric = 'w1'

    config.eval = ev = ml_collections.ConfigDict()
    ev.batch_size = 128
    ev.num_samples = 500
    ev.use_ema = True

    return config
