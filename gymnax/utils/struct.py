from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import field as _field
from dataclasses import fields as _fields
from dataclasses import replace as _dc_replace
from typing import Any, Callable, Iterable, Tuple

import jax


def dataclass(_cls: type | None = None, *, frozen: bool = True):
    """A lightweight JAX-friendly dataclass.

    - Wraps stdlib dataclass (default frozen=True)
    - Registers the class as a JAX PyTree
    - Adds an instance method `.replace(**updates)` similar to flax.struct
    - Supports marking static fields via `metadata={"pytree_node": False}`
    """

    def wrap(cls: type):
        dc = _dataclass(frozen=frozen)(cls)
        dc_fields = _fields(dc)
        names = tuple(f.name for f in dc_fields)
        pytree_mask = tuple(bool(f.metadata.get("pytree_node", True)) for f in dc_fields)

        def flatten(obj: Any) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
            children = []
            static = []
            for name, is_node in zip(names, pytree_mask):
                val = getattr(obj, name)
                if is_node:
                    children.append(val)
                else:
                    static.append(val)
            # Use names and mask to reconstruct reliably
            aux = (names, pytree_mask, tuple(static))
            return tuple(children), aux

        def unflatten(aux: Tuple[Iterable[str], Iterable[bool], Tuple[Any, ...]],
                      children: Tuple[Any, ...]):
            names_aux, mask_aux, static_vals = aux
            out_kwargs = {}
            c_it = iter(children)
            s_it = iter(static_vals)
            for name, is_node in zip(names_aux, mask_aux):
                out_kwargs[name] = next(c_it) if is_node else next(s_it)
            return dc(**out_kwargs)

        # Register as PyTree once per class
        jax.tree_util.register_pytree_node(dc, flatten, unflatten)

        # Attach replace method for convenience
        def _replace(self, **updates):
            return _dc_replace(self, **updates)

        setattr(dc, "replace", _replace)
        return dc

    if _cls is None:
        return wrap
    return wrap(_cls)


# Re-export field so code can do `from gymnax.utils import struct; struct.field(...)`
field = _field
