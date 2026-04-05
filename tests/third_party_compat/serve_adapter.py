"""Generic adapter: imports an OAI example agent and exposes it for serve.

serve needs ``module:attr`` import paths.  This module reads env vars to
decide which example to import and which Agent variable to expose::

    SERVE_EXAMPLE_MODULE=tools
    SERVE_AGENT_VAR=agent              # default: "agent"

The examples already import from ``motus.openai_agents`` so no import
hook is needed.

The actual import is deferred to first attribute access so that forkserver
preloading doesn't trigger the full import chain (litellm -> dotenv etc.).
"""

from __future__ import annotations

import importlib
import os
import sys
import types


def _load_agent():
    """Import the configured example module and return the agent."""
    module_name = os.environ.get("SERVE_EXAMPLE_MODULE", "")
    agent_var = os.environ.get("SERVE_AGENT_VAR", "agent")

    if not module_name:
        raise RuntimeError("SERVE_EXAMPLE_MODULE env var not set")

    # Ensure the examples directory is importable
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    examples_dir = os.path.join(repo_root, "examples", "openai_agents")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)

    mod = importlib.import_module(module_name)
    return getattr(mod, agent_var)


class _LazyModule(types.ModuleType):
    """Module subclass that defers agent loading to first attribute access."""

    _agent = None
    _loaded = False

    def __getattr__(self, name):
        if name == "agent" and not self._loaded:
            self._loaded = True
            self._agent = _load_agent()
            return self._agent
        if name == "agent":
            return self._agent
        raise AttributeError(name)


# Replace this module with the lazy version so `getattr(mod, 'agent')`
# triggers the import only when the worker actually needs it.
_self = sys.modules[__name__]
_lazy = _LazyModule(__name__)
_lazy.__dict__.update({k: v for k, v in _self.__dict__.items() if k.startswith("_")})
_lazy.__file__ = __file__
_lazy.__path__ = []
_lazy.__package__ = __package__
sys.modules[__name__] = _lazy
