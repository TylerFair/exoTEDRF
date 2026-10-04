"""Run calibration steps and retain intermediate states for optimizer trials."""

import dataclasses
import os
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np

from exotedrf.v2 import core


@dataclasses.dataclass
class PipelineState:
    """Store an observation and its auxiliary reduction data.

    Parameters
    ----------
    cube : RampCube, RateCube
        Observation data and metadata.
    aux : dict
        Trace positions, background models and other step products.
    """
    cube: object
    aux: dict = dataclasses.field(default_factory=dict)

    def snapshot(self, storage='host', device=None):
        """Copy the observation and all auxiliary arrays.

        Parameters
        ----------
        storage : str
            Store independent copies in host or device memory.
        device : None, jax.Device
            JAX device for accelerator snapshots.

        Returns
        -------
        state : PipelineState
            Independent copy in the selected storage.
        """
        if storage not in ('host', 'device'):
            raise ValueError("storage must be 'host' or 'device'")

        copied = {}

        def convert(x):
            """Copy each distinct array to the selected storage."""
            if not hasattr(x, 'shape'):
                return x
            key = id(x)
            if key not in copied:
                if storage == 'device':
                    try:
                        copied[key] = jnp.array(x, copy=True, device=device)
                    except (TypeError, ValueError):
                        copied[key] = core.copy_host_array(x, name='exotedrf-checkpoint')
                else:
                    copied[key] = core.copy_host_array(x, name='exotedrf-checkpoint')
            return copied[key]

        cube = jax.tree.map(convert, self.cube)
        # Copy arrays inside nested auxiliary products.
        aux = jax.tree.map(convert, self.aux)
        return PipelineState(cube=cube, aux=aux)

    def readonly_view(self):
        """Borrow arrays through read-only views and independent containers.

        The owner must not modify source arrays while the borrowed state is in use.

        Returns
        -------
        state : PipelineState
            State sharing the source arrays through read-only views.
        """
        views = {}

        def borrow(value):
            """Create one read-only view for each distinct host array."""
            if not hasattr(value, 'shape') or isinstance(value, jax.Array):
                return value
            key = id(value)
            if key not in views:
                if isinstance(value, np.ndarray):
                    view = value.view()
                    view.setflags(write=False)
                elif isinstance(value, np.generic):
                    view = value
                else:
                    raise TypeError('read-only checkpoints require NumPy or JAX arrays')
                views[key] = view
            return views[key]
        return PipelineState(cube=jax.tree.map(borrow, self.cube),
                              aux=jax.tree.map(borrow, self.aux))


@dataclasses.dataclass
class Step:
    """Describe a named reduction step and its parameter dependencies.

    Parameters
    ----------
    name : str
        Reduction step name.
    fn : callable
        Function returning a new state from state, params and ctx.
    param_names, modes : tuple[str]
        Consumed parameters and applicable mode prefixes (empty accepts every mode).
    """
    name: str
    fn: object
    param_names: tuple = ()
    modes: tuple = ()

    def applies(self, mode):
        """Check whether the step applies to an observing mode.

        Parameters
        ----------
        mode : str
            Instrument and observing mode.

        Returns
        -------
        applies : bool
            Whether the mode matches the step restrictions.
        """
        return not self.modes or any(mode.upper().startswith(m.upper()) for m in self.modes)


class CheckpointStore:
    """Retain states immediately before selected reduction steps.

    Parameters
    ----------
    keep : None, list[str]
        Step names to retain. If None, retain every step.
    storage : str
        Use host, device, device_if_fits or borrowed_readonly storage.
    max_device_bytes : None, int
        Maximum total accelerator storage for saved states.
    device : None, jax.Device
        JAX device for accelerator snapshots.
    """

    _STORAGE_POLICIES = frozenset(('host', 'device', 'device_if_fits', 'borrowed_readonly'))

    def __init__(self, keep=None, storage='host', max_device_bytes=None, device=None):
        """Initialize the checkpoint store."""
        if storage not in self._STORAGE_POLICIES:
            raise ValueError(f'storage must be one of {sorted(self._STORAGE_POLICIES)}')
        if storage == 'device_if_fits' and max_device_bytes is None:
            raise ValueError('max_device_bytes is required for device_if_fits storage')
        if storage == 'borrowed_readonly' and keep is None:
            raise ValueError('borrowed checkpoints require explicit step names')
        if max_device_bytes is not None and max_device_bytes < 0:
            raise ValueError('max_device_bytes must be non-negative')
        self._snaps = {}
        self._locations = {}
        self._sizes = {}
        self.keep = set(keep) if keep is not None else None
        self.storage = storage
        self.max_device_bytes = max_device_bytes
        self.device = device

    @classmethod
    def device_resident(cls, keep=None, device=None):
        """Create a store for accelerator snapshots.

        Parameters
        ----------
        keep : None, list[str]
            Step names to retain. If None, retain every step.
        device : None, jax.Device
            JAX device for accelerator snapshots.

        Returns
        -------
        store : CheckpointStore
            Store retaining the requested states on the accelerator.
        """
        return cls(keep=keep, storage='device', device=device)

    @classmethod
    def device_if_fits(cls, max_device_bytes, keep=None, device=None):
        """Create a store that uses accelerator memory up to a limit.

        Parameters
        ----------
        max_device_bytes : None, int
            Maximum total accelerator storage for saved states.
        keep : None, list[str]
            Step names to retain. If None, retain every step.
        device : None, jax.Device
            JAX device for accelerator snapshots.

        Returns
        -------
        store : CheckpointStore
            Store falling back to host snapshots when the limit is reached.
        """
        return cls(keep=keep, storage='device_if_fits',
                   max_device_bytes=max_device_bytes, device=device)

    @classmethod
    def borrowed_readonly(cls, keep):
        """Create a store for explicitly selected borrowed states.

        The owner and downstream steps must not modify the source arrays.

        Parameters
        ----------
        keep : list[str]
            Explicit step names to retain through read-only views.

        Returns
        -------
        store : CheckpointStore
            Store retaining read-only views of the requested states.
        """
        if keep is None:
            raise ValueError('borrowed checkpoints require explicit step names')
        return cls(keep=keep, storage='borrowed_readonly')

    @staticmethod
    def _state_nbytes(state):
        """Count distinct array bytes in a pipeline state."""
        total = 0
        seen = set()
        for leaf in jax.tree.leaves(state.cube) + jax.tree.leaves(state.aux):
            if hasattr(leaf, 'nbytes') and id(leaf) not in seen:
                total += leaf.nbytes
                seen.add(id(leaf))
        return total

    def _storage_for(self, name, nbytes):
        """Choose snapshot storage within the accelerator memory allowance."""
        if self.storage != 'device_if_fits':
            return self.storage
        # Subtract the replaced checkpoint from the accelerator memory usage.
        used = self.device_nbytes()
        if self._locations.get(name) == 'device':
            used -= self._sizes[name]
        return ('device' if used + nbytes <= self.max_device_bytes else 'host')

    def put(self, name, state):
        """Save a state immediately before a named step.

        Parameters
        ----------
        name : str
            Reduction step name.
        state : PipelineState
            Observation and auxiliary data.
        """
        if self.keep is None or name in self.keep:
            if self._snaps.get(name) is state:
                # Reuse the independent snapshot when committing from that state.
                return
            nbytes = self._state_nbytes(state)
            location = self._storage_for(name, nbytes)
            self._snaps[name] = (state.readonly_view() if location == 'borrowed_readonly' else
                state.snapshot(storage=location, device=self.device))
            self._locations[name] = location
            self._sizes[name] = nbytes

    def get(self, name):
        """Get the saved state for a named step."""
        return self._snaps[name]

    def __contains__(self, name):
        """Check whether a named checkpoint is saved."""
        return name in self._snaps

    def drop(self, name):
        """Release a saved state and its storage metadata.

        Parameters
        ----------
        name : str
            Reduction step name.
        """
        self._snaps.pop(name, None)
        self._locations.pop(name, None)
        self._sizes.pop(name, None)

    def location(self, name):
        """Get the storage policy of a saved state.

        Parameters
        ----------
        name : str
            Reduction step name.

        Returns
        -------
        location : str
            host, device or borrowed_readonly.
        """
        return self._locations[name]

    def nbytes(self):
        """Count bytes retained by all saved states."""
        return sum(self._sizes.values())

    def device_nbytes(self):
        """Count bytes retained by accelerator snapshots."""
        return sum(size for name, size in self._sizes.items() if self._locations[name] == 'device')

    def host_nbytes(self):
        """Count bytes owned by host snapshots."""
        return self.nbytes() - self.device_nbytes() - self.borrowed_nbytes()

    def borrowed_nbytes(self):
        """Count bytes retained through borrowed arrays."""
        return sum(size for name, size in self._sizes.items()
                   if self._locations[name] == 'borrowed_readonly')


class Pipeline:
    """Run the ordered calibration recipe for one observing mode.

    Parameters
    ----------
    steps : list[Step]
        Reduction steps in execution order.
    mode : str
        Instrument and observing mode.
    ctx : dict
        Shared reduction context.
    """

    def __init__(self, steps, mode, ctx):
        """Initialize the observing-mode pipeline."""
        self.steps = [s for s in steps if s.applies(mode)]
        self.mode = mode
        self.ctx = ctx
        self._index = {s.name: i for i, s in enumerate(self.steps)}

    def index_of(self, name):
        """Get the index of a named reduction step."""
        return self._index[name]

    def first_consumer(self, param_name):
        """Find the first step affected by a reduction parameter.

        Parameters
        ----------
        param_name : str
            Reduction parameter name.

        Returns
        -------
        index : int
            Index of the first step consuming the parameter.
        """
        for i, s in enumerate(self.steps):
            if param_name in s.param_names:
                return i
        raise KeyError(f'No step consumes parameter {param_name!r}')

    def run(self, state, params, start=0, stop=None, store=None, verbose=False, observer=None):
        """Run consecutive reduction steps and save requested checkpoints.

        Parameters
        ----------
        state : PipelineState
            Observation and auxiliary data.
        params : dict
            Reduction parameter values.
        start : int
            Index of the first step to run.
        stop : None, int
            Exclusive index of the last step. If None, run through the final step.
        store : None, CheckpointStore
            Store for states immediately before the selected steps.
        verbose : bool
            If True, print each step name.
        observer : None, callable
            Function called with the step and its input and output states.

        Returns
        -------
        state : PipelineState
            State after the selected steps.
        """
        stop = len(self.steps) if stop is None else stop
        profile = os.environ.get('EXOTEDRF_PROFILE_STEPS', '').lower() in ('1', 'true', 'yes', 'on')
        for i in range(start, stop):
            step = self.steps[i]
            if store is not None:
                store.put(step.name, state)
            if verbose:
                print(f'[v2] running {step.name}')
            started = time.perf_counter() if profile else None
            usage_before = (resource.getrusage(resource.RUSAGE_SELF) if profile else None)
            before = state
            state = step.fn(state, params, self.ctx)
            if observer is not None:
                observer(step, before, state)
            # Release the previous cube before saving the next checkpoint.
            del before
            if profile:
                # Wait for accelerator calculations before reporting step timings.
                for leaf in (jax.tree.leaves(state.cube) + jax.tree.leaves(state.aux)):
                    ready = getattr(leaf, 'block_until_ready', None)
                    if ready is not None:
                        ready()
                duration = time.perf_counter() - started
                usage = resource.getrusage(resource.RUSAGE_SELF)
                read = (usage.ru_inblock - usage_before.ru_inblock) * 512
                written = (usage.ru_oublock - usage_before.ru_oublock) * 512
                print(f'[v2-profile] {step.name}: {duration:.3f}s; '
                      f'filesystem read={read / 2**30:.3f} GiB ' f'write={written / 2**30:.3f} GiB')
        return state

    def rerun_for(self, changed_params, params, store, stop=None):
        """Evaluate changed parameters from the first affected checkpoint.

        Parameters
        ----------
        changed_params : list[str]
            Parameters changed since the saved reduction.
        params : dict
            Reduction parameter values.
        store : CheckpointStore
            Saved states immediately before reduction steps.
        stop : None, int
            Exclusive index of the last step. If None, run through the final step.

        Returns
        -------
        state : PipelineState
            Trial state with saved checkpoints left unchanged.
        """
        start = min(self.first_consumer(p) for p in changed_params)
        state = store.get(self.steps[start].name)
        return self.run(state, params, start=start, stop=stop, store=None)

    def commit_from(self, changed_params, params, store, stop=None):
        """Apply winning parameters and refresh downstream checkpoints.

        Parameters
        ----------
        changed_params : list[str]
            Parameters changed since the saved reduction.
        params : dict
            Reduction parameter values.
        store : CheckpointStore
            Saved states immediately before reduction steps.
        stop : None, int
            Exclusive index of the last step. If None, run through the final step.

        Returns
        -------
        state : PipelineState
            Committed state after the selected steps.
        """
        start = min(self.first_consumer(p) for p in changed_params)
        state = store.get(self.steps[start].name)
        return self.run(state, params, start=start, stop=stop, store=store)
