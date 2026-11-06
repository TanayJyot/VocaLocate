"""Explicit architecture selection for controlled pretraining experiments."""

EXPERIMENTS = ("baseline", "looped", "diff-v2", "combined")


def select_experiment(args):
    """Resolve an explicit selection, rejecting flags for other architectures.

    Omitting the selector preserves existing CLI calls. ``combined`` requires
    both ablations and makes the intention to mix them explicit.
    """
    selected = args.experiment
    if selected is None:
        return
    active = set()
    if args.n_loops != 1 or getattr(args, "loop_embed", False):
        active.add("looped")
    if getattr(args, "diff_v2", False):
        active.add("diff-v2")
    if selected == "combined":
        if len(active) < 2:
            raise ValueError("--experiment=combined requires at least two enabled ablations")
        return
    conflicts = active - {selected}
    if conflicts:
        raise ValueError(f"--experiment={selected} conflicts with enabled ablations: {', '.join(sorted(conflicts))}")
    if selected != "looped" and (args.n_prelude != 0 or args.n_coda != 0):
        raise ValueError(f"--experiment={selected} requires --n-prelude=0 and --n-coda=0")
    if selected == "looped":
        if args.n_loops < 2:
            raise ValueError("--experiment=looped requires --n-loops >= 2")
    elif selected == "diff-v2":
        args.diff_v2 = True
