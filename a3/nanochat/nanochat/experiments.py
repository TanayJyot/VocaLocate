"""Explicit architecture selection for controlled pretraining experiments."""

EXPERIMENTS = ("baseline", "looped", "xiprelu", "diff-v2", "depth-attention", "mgvr", "xsa", "mudd", "combined")


def select_experiment(args):
    """Resolve an explicit selection, rejecting flags for other architectures.

    Omitting the selector preserves existing CLI calls. ``combined`` requires
    at least two enabled ablations and makes the intention to mix them explicit.
    """
    selected = args.experiment
    if selected is None:
        return
    active = set()
    if args.n_loops != 1 or any(getattr(args, k, 0) for k in ("loop_embed", "n_loops_min", "n_loops_max")):
        active.add("looped")
    if args.mlp_activation != "relu2":
        active.add("xiprelu")
    for name in ("diff-v2", "depth-attention", "mgvr", "xsa", "mudd"):
        if getattr(args, name.replace("-", "_"), False):
            active.add(name)
    if selected == "combined":
        if len(active) < 2:
            raise ValueError("--experiment=combined requires at least two enabled ablations")
        return
    conflicts = active - {selected}
    if conflicts:
        raise ValueError(f"--experiment={selected} conflicts with enabled ablations: {', '.join(sorted(conflicts))}")
    if selected != "looped" and (args.n_prelude != 0 or args.n_coda != 0):
        raise ValueError(f"--experiment={selected} requires --n-prelude=0 and --n-coda=0")
    if selected != "depth-attention" and args.depth_attention_stride is not None:
        raise ValueError(f"--experiment={selected} cannot set --depth-attention-stride")
    if selected == "looped":
        if args.n_loops < 2:
            raise ValueError("--experiment=looped requires --n-loops >= 2")
    elif selected == "xiprelu":
        args.mlp_activation = "xiprelu"
    elif selected != "baseline":
        setattr(args, selected.replace("-", "_"), True)
