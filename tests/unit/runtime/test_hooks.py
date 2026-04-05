import sys
import unittest
from unittest import mock

from motus.runtime.hooks import (
    HookEvent,
    HookManager,
    global_hook,
    model_task_hook,
    register_hook,
    register_task_hook,
    register_type_hook,
    task_hook,
    tool_task_hook,
    type_hook,
)

# ``from motus.runtime.hooks import hooks`` returns the HookManager instance.
# We need the *module* to swap out the singleton during tests.
_hooks_module = sys.modules["motus.runtime.hooks"]
_runtime_module = sys.modules.get("motus.runtime.agent_runtime")


def _make_event(event_type="task_start", name=None, **kwargs):
    return HookEvent(event_type=event_type, name=name, **kwargs)


class _SwapHooksMixin:
    """Replace the module-level ``hooks`` singleton with a fresh HookManager
    for the duration of each test, then restore the original.

    Also patches the reference held by ``agent_runtime`` so that
    ``@agent_task(on_start=...)`` writes into the same fresh instance.
    """

    def setUp(self):
        global _runtime_module
        if _runtime_module is None:
            import motus.runtime.agent_runtime  # noqa: F401

            _runtime_module = sys.modules["motus.runtime.agent_runtime"]

        self._orig = _hooks_module.hooks
        fresh = HookManager()
        _hooks_module.hooks = fresh
        self._rt_orig = _runtime_module.hooks
        _runtime_module.hooks = fresh

    def tearDown(self):
        _hooks_module.hooks = self._orig
        _runtime_module.hooks = self._rt_orig


# ── HookManager unit tests ───────────────────────────────────────


class TestHookManagerGlobal(unittest.IsolatedAsyncioTestCase):
    """Global hook registration and emission."""

    def setUp(self):
        self.mgr = HookManager()

    async def test_register_and_emit_sync(self):
        called = []
        self.mgr.register("task_start", lambda e: called.append(e))
        event = _make_event("task_start", name="foo")
        await self.mgr.emit(event)
        self.assertEqual(called, [event])

    async def test_register_and_emit_async(self):
        called = []

        async def cb(e):
            called.append(e)

        self.mgr.register("task_end", cb)
        event = _make_event("task_end", name="foo")
        await self.mgr.emit(event)
        self.assertEqual(called, [event])

    async def test_emit_no_hooks_does_not_raise(self):
        await self.mgr.emit(_make_event("task_error", name="x"))

    async def test_multiple_callbacks_fire_in_order(self):
        order = []
        self.mgr.register("task_start", lambda e: order.append("a"))
        self.mgr.register("task_start", lambda e: order.append("b"))
        await self.mgr.emit(_make_event("task_start", name="t"))
        self.assertEqual(order, ["a", "b"])

    async def test_prepend_puts_callback_first(self):
        order = []
        self.mgr.register("task_start", lambda e: order.append("a"))
        self.mgr.register("task_start", lambda e: order.append("b"), prepend=True)
        await self.mgr.emit(_make_event("task_start", name="t"))
        self.assertEqual(order, ["b", "a"])

    async def test_global_fires_for_any_task_name(self):
        called = []
        self.mgr.register("task_start", lambda e: called.append(e.name))
        await self.mgr.emit(_make_event("task_start", name="alpha"))
        await self.mgr.emit(_make_event("task_start", name="beta"))
        self.assertEqual(called, ["alpha", "beta"])

    async def test_callback_exception_does_not_propagate(self):
        ok = []

        def bad(e):
            raise RuntimeError("boom")

        self.mgr.register("task_start", bad)
        self.mgr.register("task_start", lambda e: ok.append(True))
        await self.mgr.emit(_make_event("task_start", name="t"))
        self.assertEqual(ok, [True])

    def test_list_hooks(self):
        def cb1(e):
            return None

        def cb2(e):
            return None

        self.mgr.register("task_start", cb1)
        self.mgr.register("task_start", cb2)
        self.assertEqual(list(self.mgr.list_hooks("task_start")), [cb1, cb2])
        self.assertEqual(list(self.mgr.list_hooks("task_end")), [])


class TestHookManagerPerName(unittest.IsolatedAsyncioTestCase):
    """Per-name hook registration and emission."""

    def setUp(self):
        self.mgr = HookManager()

    async def test_name_hook_fires_only_for_matching_name(self):
        called = []
        self.mgr.register_name_hook(
            "foo", "task_start", lambda e: called.append(e.name)
        )
        await self.mgr.emit(_make_event("task_start", name="foo"))
        await self.mgr.emit(_make_event("task_start", name="bar"))
        self.assertEqual(called, ["foo"])

    async def test_name_hook_accepts_callable(self):
        called = []

        def my_func():
            pass

        self.mgr.register_name_hook(
            my_func, "task_start", lambda e: called.append(True)
        )
        await self.mgr.emit(_make_event("task_start", name=my_func.__qualname__))
        self.assertEqual(called, [True])

    async def test_name_hook_prepend(self):
        order = []
        self.mgr.register_name_hook("t", "task_start", lambda e: order.append("a"))
        self.mgr.register_name_hook(
            "t", "task_start", lambda e: order.append("b"), prepend=True
        )
        await self.mgr.emit(_make_event("task_start", name="t"))
        self.assertEqual(order, ["b", "a"])

    async def test_name_hook_async_callback(self):
        called = []

        async def cb(e):
            called.append(e.name)

        self.mgr.register_name_hook("t", "task_end", cb)
        await self.mgr.emit(_make_event("task_end", name="t"))
        self.assertEqual(called, ["t"])

    def test_list_name_hooks(self):
        def cb(e):
            return None

        self.mgr.register_name_hook("t", "task_start", cb)
        self.assertEqual(list(self.mgr.list_name_hooks("t", "task_start")), [cb])
        self.assertEqual(list(self.mgr.list_name_hooks("t", "task_end")), [])
        self.assertEqual(list(self.mgr.list_name_hooks("other", "task_start")), [])

    async def test_event_without_name_skips_name_hooks(self):
        called = []
        self.mgr.register_name_hook("t", "task_start", lambda e: called.append(True))
        await self.mgr.emit(_make_event("task_start", name=None))
        self.assertEqual(called, [])

    async def test_name_hook_exception_does_not_propagate(self):
        ok = []

        def bad(e):
            raise RuntimeError("boom")

        self.mgr.register_name_hook("t", "task_start", bad)
        self.mgr.register_name_hook("t", "task_start", lambda e: ok.append(True))
        await self.mgr.emit(_make_event("task_start", name="t"))
        self.assertEqual(ok, [True])


class TestHookManagerPerType(unittest.IsolatedAsyncioTestCase):
    """Per-type hook registration and emission."""

    def setUp(self):
        self.mgr = HookManager()

    async def test_type_hook_fires_only_for_matching_type(self):
        called = []
        self.mgr.register_type_hook(
            "tool_call", "task_end", lambda e: called.append(e.name)
        )
        await self.mgr.emit(
            _make_event("task_end", name="web_search", task_type="tool_call")
        )
        await self.mgr.emit(
            _make_event("task_end", name="model_serve_task", task_type="model_call")
        )
        self.assertEqual(called, ["web_search"])

    async def test_type_hook_fires_for_all_tasks_of_that_type(self):
        called = []
        self.mgr.register_type_hook(
            "tool_call", "task_end", lambda e: called.append(e.name)
        )
        await self.mgr.emit(
            _make_event("task_end", name="web_search", task_type="tool_call")
        )
        await self.mgr.emit(
            _make_event("task_end", name="calculator", task_type="tool_call")
        )
        self.assertEqual(called, ["web_search", "calculator"])

    def test_list_type_hooks(self):
        def cb(e):
            return None

        self.mgr.register_type_hook("tool_call", "task_start", cb)
        self.assertEqual(
            list(self.mgr.list_type_hooks("tool_call", "task_start")), [cb]
        )
        self.assertEqual(list(self.mgr.list_type_hooks("tool_call", "task_end")), [])
        self.assertEqual(list(self.mgr.list_type_hooks("model_call", "task_start")), [])

    async def test_event_without_type_skips_type_hooks(self):
        called = []
        self.mgr.register_type_hook(
            "tool_call", "task_start", lambda e: called.append(True)
        )
        await self.mgr.emit(_make_event("task_start", name="foo"))
        self.assertEqual(called, [])


class TestHookManagerEmitOrder(unittest.IsolatedAsyncioTestCase):
    """Global hooks fire before name hooks, then type hooks."""

    def setUp(self):
        self.mgr = HookManager()

    async def test_global_before_name(self):
        order = []
        self.mgr.register("task_start", lambda e: order.append("global"))
        self.mgr.register_name_hook("t", "task_start", lambda e: order.append("name"))
        await self.mgr.emit(_make_event("task_start", name="t"))
        self.assertEqual(order, ["global", "name"])

    async def test_global_before_name_before_type(self):
        order = []
        self.mgr.register("task_end", lambda e: order.append("global"))
        self.mgr.register_name_hook(
            "web_search", "task_end", lambda e: order.append("name")
        )
        self.mgr.register_type_hook(
            "tool_call", "task_end", lambda e: order.append("type")
        )
        await self.mgr.emit(
            _make_event("task_end", name="web_search", task_type="tool_call")
        )
        self.assertEqual(order, ["global", "name", "type"])

    async def test_mixed_event_types(self):
        starts = []
        ends = []
        self.mgr.register("task_start", lambda e: starts.append("g"))
        self.mgr.register_name_hook("t", "task_start", lambda e: starts.append("n"))
        self.mgr.register("task_end", lambda e: ends.append("g"))
        self.mgr.register_name_hook("t", "task_end", lambda e: ends.append("n"))
        await self.mgr.emit(_make_event("task_start", name="t"))
        await self.mgr.emit(_make_event("task_end", name="t"))
        self.assertEqual(starts, ["g", "n"])
        self.assertEqual(ends, ["g", "n"])


# ── Module-level convenience functions / decorators ───────────────


class TestModuleLevelRegisterHook(_SwapHooksMixin, unittest.IsolatedAsyncioTestCase):
    async def test_register_hook(self):
        called = []
        register_hook("task_start", lambda e: called.append(True))
        await _hooks_module.hooks.emit(_make_event("task_start", name="x"))
        self.assertEqual(called, [True])

    async def test_register_task_hook_str(self):
        called = []
        register_task_hook("x", "task_end", lambda e: called.append(True))
        await _hooks_module.hooks.emit(_make_event("task_end", name="x"))
        await _hooks_module.hooks.emit(_make_event("task_end", name="y"))
        self.assertEqual(called, [True])

    async def test_register_task_hook_callable(self):
        called = []

        def my_task():
            pass

        register_task_hook(my_task, "task_start", lambda e: called.append(True))
        await _hooks_module.hooks.emit(
            _make_event("task_start", name=my_task.__qualname__)
        )
        self.assertEqual(called, [True])

    async def test_register_type_hook(self):
        called = []
        register_type_hook("tool_call", "task_end", lambda e: called.append(True))
        await _hooks_module.hooks.emit(
            _make_event("task_end", name="web_search", task_type="tool_call")
        )
        self.assertEqual(called, [True])


class TestGlobalHookDecorator(_SwapHooksMixin, unittest.IsolatedAsyncioTestCase):
    async def test_global_hook_decorator(self):
        called = []

        @global_hook("task_start")
        def cb(e):
            called.append(e.name)

        await _hooks_module.hooks.emit(_make_event("task_start", name="abc"))
        self.assertEqual(called, ["abc"])

    async def test_global_hook_decorator_returns_original_function(self):
        @global_hook("task_start")
        def cb(e):
            return 42

        self.assertEqual(cb(_make_event("task_start")), 42)


class TestTaskHookDecorator(_SwapHooksMixin, unittest.IsolatedAsyncioTestCase):
    async def test_task_hook_decorator_str(self):
        called = []

        @task_hook("my_task", "task_start")
        def cb(e):
            called.append(True)

        await _hooks_module.hooks.emit(_make_event("task_start", name="my_task"))
        await _hooks_module.hooks.emit(_make_event("task_start", name="other"))
        self.assertEqual(called, [True])

    async def test_task_hook_decorator_callable(self):
        called = []

        def target_fn():
            pass

        @task_hook(target_fn, "task_end")
        def cb(e):
            called.append(True)

        await _hooks_module.hooks.emit(
            _make_event("task_end", name=target_fn.__qualname__)
        )
        self.assertEqual(called, [True])

    async def test_task_hook_decorator_returns_original_function(self):
        @task_hook("t", "task_start")
        def cb(e):
            return 99

        self.assertEqual(cb(_make_event("task_start")), 99)


class TestTypeHookDecorator(_SwapHooksMixin, unittest.IsolatedAsyncioTestCase):
    async def test_type_hook_decorator(self):
        called = []

        @type_hook("tool_call", "task_end")
        def cb(e):
            called.append(e.name)

        await _hooks_module.hooks.emit(
            _make_event("task_end", name="web_search", task_type="tool_call")
        )
        self.assertEqual(called, ["web_search"])

    async def test_type_hook_ignores_other_types(self):
        called = []

        @type_hook("tool_call", "task_end")
        def cb(e):
            called.append(True)

        await _hooks_module.hooks.emit(
            _make_event("task_end", name="model_serve_task", task_type="model_call")
        )
        self.assertEqual(called, [])


class TestModelTaskHookDecorator(_SwapHooksMixin, unittest.IsolatedAsyncioTestCase):
    async def test_fires_for_model_call_type(self):
        called = []

        @model_task_hook("task_start")
        def cb(e):
            called.append(e.name)

        await _hooks_module.hooks.emit(
            _make_event("task_start", name="model_serve_task", task_type="model_call")
        )
        self.assertEqual(called, ["model_serve_task"])

    async def test_ignores_other_types(self):
        called = []

        @model_task_hook("task_start")
        def cb(e):
            called.append(True)

        await _hooks_module.hooks.emit(
            _make_event("task_start", name="web_search", task_type="tool_call")
        )
        await _hooks_module.hooks.emit(_make_event("task_start", name="random"))
        self.assertEqual(called, [])


class TestToolTaskHookDecorator(_SwapHooksMixin, unittest.IsolatedAsyncioTestCase):
    async def test_fires_for_tool_call_type(self):
        called = []

        @tool_task_hook("task_end")
        def cb(e):
            called.append(e.name)

        await _hooks_module.hooks.emit(
            _make_event("task_end", name="web_search", task_type="tool_call")
        )
        self.assertEqual(called, ["web_search"])

    async def test_fires_for_all_tools(self):
        """tool_task_hook fires for any event with task_type='tool_call'."""
        called = []

        @tool_task_hook("task_end")
        def cb(e):
            called.append(e.name)

        await _hooks_module.hooks.emit(
            _make_event("task_end", name="web_search", task_type="tool_call")
        )
        await _hooks_module.hooks.emit(
            _make_event("task_end", name="calculator", task_type="tool_call")
        )
        self.assertEqual(called, ["web_search", "calculator"])

    async def test_ignores_other_types(self):
        called = []

        @tool_task_hook("task_end")
        def cb(e):
            called.append(True)

        await _hooks_module.hooks.emit(
            _make_event("task_end", name="model_serve_task", task_type="model_call")
        )
        await _hooks_module.hooks.emit(_make_event("task_end", name="random"))
        self.assertEqual(called, [])


# ── @agent_task decorator hook registration ───────────────────────


class TestAgentTaskHookRegistration(unittest.TestCase):
    """@agent_task(on_start=..., on_end=..., on_error=...) registers per-name hooks."""

    def test_no_args_registers_with_none_values(self):
        from motus.runtime import agent_task

        with mock.patch(
            "motus.runtime.agent_task._register_task_hooks"
        ) as mock_register:

            @agent_task
            def my_fn():
                pass

            mock_register.assert_called_once_with(mock.ANY, None, None, None)

    def test_with_single_callbacks(self):
        from motus.runtime import agent_task

        def cb_start(e):
            return None

        def cb_end(e):
            return None

        def cb_err(e):
            return None

        with mock.patch(
            "motus.runtime.agent_task._register_task_hooks"
        ) as mock_register:

            @agent_task(on_start=cb_start, on_end=cb_end, on_error=cb_err)
            def my_fn():
                pass

            mock_register.assert_called_once()
            name, start, end, err = mock_register.call_args[0]
            self.assertTrue(name.endswith(".my_fn"))
            self.assertEqual(start, cb_start)
            self.assertEqual(end, cb_end)
            self.assertEqual(err, cb_err)

    def test_with_callback_lists(self):
        from motus.runtime import agent_task

        def cb1(e):
            return None

        def cb2(e):
            return None

        with mock.patch(
            "motus.runtime.agent_task._register_task_hooks"
        ) as mock_register:

            @agent_task(on_end=[cb1, cb2])
            def my_fn():
                pass

            mock_register.assert_called_once()
            name, start, end, err = mock_register.call_args[0]
            self.assertTrue(name.endswith(".my_fn"))
            self.assertIsNone(start)
            self.assertEqual(end, [cb1, cb2])
            self.assertIsNone(err)

    def test_preserves_name(self):
        from motus.runtime import agent_task

        @agent_task
        def some_task():
            pass

        self.assertEqual(some_task.__name__, "some_task")

        @agent_task(on_start=lambda e: None)
        def another_task():
            pass

        self.assertEqual(another_task.__name__, "another_task")


# ── HookEvent dataclass ──────────────────────────────────────────


class TestHookEvent(unittest.TestCase):
    def test_defaults(self):
        e = HookEvent(event_type="task_start")
        self.assertIsNone(e.name)
        self.assertIsNone(e.args)
        self.assertIsNone(e.kwargs)
        self.assertIsNone(e.result)
        self.assertIsNone(e.error)
        self.assertIsNone(e.task_id)
        self.assertEqual(e.metadata, {})

    def test_frozen(self):
        e = HookEvent(event_type="task_start", name="t")
        with self.assertRaises(AttributeError):
            e.name = "changed"

    def test_metadata(self):
        e = HookEvent(event_type="task_start", metadata={"key": "val"})
        self.assertEqual(e.metadata["key"], "val")


if __name__ == "__main__":
    unittest.main()
