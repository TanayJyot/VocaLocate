"""CPU numerical checks of distributed AdamW with simulated async collectives.

Real parameter slices and optimizer updates run on every simulated rank; only
the transport is replaced. NCCL execution still requires a multi-GPU check.
"""
import copy
from types import SimpleNamespace

import pytest
import torch

import nanochat.optim as optim_module
from nanochat.gpt import MGVR
from nanochat.optim import MuonAdamW


class Collectives:
    def __init__(self, world_size):
        self.world_size = world_size
        self.rank = 0
        self.pending = {}
        self.indices = {}

    def enqueue(self, kind, output, source):
        key = (self.rank, kind)
        index = self.indices.get(key, 0)
        self.indices[key] = index + 1
        entries = self.pending.setdefault((kind, index), [])
        future = torch.futures.Future()
        entries.append((self.rank, output, source.clone(), future))
        if len(entries) == self.world_size:
            entries.sort(key=lambda entry: entry[0])
            if kind == "gather":
                result = torch.cat([entry[2] for entry in entries])
            else:
                result = torch.stack([entry[2] for entry in entries]).mean(0)
            for rank, destination, _, completed in entries:
                value = result.chunk(self.world_size)[rank] if kind == "scatter" else result
                with torch.no_grad():
                    destination.copy_(value)
                completed.set_result(None)
            del self.pending[(kind, index)]
        return SimpleNamespace(get_future=lambda: future)

    def all_reduce(self, tensor, op, async_op):
        assert op == torch.distributed.ReduceOp.AVG and async_op
        return self.enqueue("reduce", tensor, tensor)

    def reduce_scatter_tensor(self, output, tensor, op, async_op):
        assert op == torch.distributed.ReduceOp.AVG and async_op
        return self.enqueue("scatter", output, tensor)

    def all_gather_into_tensor(self, output, tensor, async_op):
        assert async_op
        return self.enqueue("gather", output, tensor)


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("heads", [2, 6, 8])
def test_mgvr_distributed_adamw_matches_averaged_gradient_reference(monkeypatch, world_size, heads):
    torch.manual_seed(91)
    module = MGVR(768, heads)
    originals = [torch.randn_like(p) * 0.05 for p in module.parameters()]
    originals += [torch.randn(32, 64) * 0.05, torch.tensor(0.2)]
    hypers = dict(lr=0.008, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    parameters = [[torch.nn.Parameter(p.clone()) for p in originals] for _ in range(world_size)]
    optimizers = [MuonAdamW([dict(kind="adamw", params=params, **hypers)]) for params in parameters]
    reference_parameters = [torch.nn.Parameter(p.clone()) for p in originals]
    reference = torch.optim.AdamW(reference_parameters, **hypers)
    transport = Collectives(world_size)
    for name in ("all_reduce", "reduce_scatter_tensor", "all_gather_into_tensor"):
        monkeypatch.setattr(optim_module.dist, name, getattr(transport, name))

    for step in range(3):
        rank_gradients = []
        for rank, params in enumerate(parameters):
            generator = torch.Generator().manual_seed(100 * step + rank)
            gradients = [torch.randn(p.shape, generator=generator) for p in params]
            rank_gradients.append(gradients)
            for p, grad in zip(params, gradients):
                p.grad = grad.clone()
        for index, p in enumerate(reference_parameters):
            p.grad = torch.stack([grads[index] for grads in rank_gradients]).mean(0)
        reference.step()

        reductions, gathers = [], []
        for rank, optimizer in enumerate(optimizers):
            transport.rank = rank
            reductions.append(optimizer._reduce_adamw(optimizer.param_groups[0], world_size))
        for rank, (optimizer, info) in enumerate(zip(optimizers, reductions)):
            transport.rank = rank
            pending = []
            with torch.no_grad(), torch.compiler.set_stance("force_eager"):
                optimizer._compute_adamw(optimizer.param_groups[0], info, pending, rank, world_size)
            gathers.append(pending)
        for optimizer, pending in zip(optimizers, gathers):
            optimizer._finish_gathers(pending)
        assert not transport.pending

        for params, optimizer in zip(parameters, optimizers):
            for actual, expected in zip(params, reference_parameters):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            rows = heads + heads**2
            expected_rows = rows // world_size if rows % world_size == 0 else rows
            assert optimizer.state[params[0]]["exp_avg"].shape == (expected_rows, 768)
            assert optimizer.state[params[2]]["exp_avg"].shape == (32 // world_size, 64)
            # A full optimizer state roundtrip must preserve either storage layout.
            if step == 1:
                optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
