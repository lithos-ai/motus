"""Tests for motus.utils.scheduler — CloudScheduler and LocalScheduler."""

import asyncio
import json
import os
import unittest
from io import BytesIO
from unittest import mock

from motus.utils.scheduler import (
    CloudScheduler,
    LocalScheduler,
    SchedulerError,
    _parse_interval,
)

# ======================================================================
# Helpers
# ======================================================================


def _mock_response(body: dict | list | None = None) -> mock.MagicMock:
    """Build a fake context-manager return for urlopen."""
    raw = json.dumps(body).encode() if body is not None else b""
    resp = mock.MagicMock()
    resp.read.return_value = raw
    resp.__enter__ = mock.Mock(return_value=resp)
    resp.__exit__ = mock.Mock(return_value=False)
    return resp


# ======================================================================
# CloudScheduler
# ======================================================================


class TestCloudSchedulerDefaults(unittest.TestCase):
    def test_reads_env_vars(self):
        env = {
            "LITHOSAI_API_URL": "https://api.example.com",
            "LITHOSAI_API_KEY": "key_123",
            "MOTUS_SESSION_ID": "sess_abc",
            "SESSION_TOKEN": "ses_tok",
            "MOTUS_PROJECT": "proj_xyz",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            s = CloudScheduler()
        self.assertEqual(s._api_url, "https://api.example.com")
        self.assertEqual(s._api_key, "key_123")
        self.assertEqual(s._session_id, "sess_abc")
        self.assertEqual(s._session_token, "ses_tok")
        self.assertEqual(s._project_id, "proj_xyz")

    def test_explicit_args_override_env(self):
        with mock.patch.dict(
            os.environ, {"LITHOSAI_API_URL": "https://wrong.com"}, clear=False
        ):
            s = CloudScheduler(api_url="https://right.com")
        self.assertEqual(s._api_url, "https://right.com")

    def test_strips_trailing_slash(self):
        s = CloudScheduler(api_url="https://api.example.com/")
        self.assertEqual(s._api_url, "https://api.example.com")


class TestCloudSchedulerSchedule(unittest.IsolatedAsyncioTestCase):
    async def test_sends_post_with_correct_body(self):
        fake_resp = _mock_response({"notification_id": "notif_001", "status": "active"})

        s = CloudScheduler(
            api_url="https://api.example.com",
            api_key="key_123",
            session_id="sess_abc",
            session_token="ses_tok",
            project_id="proj_xyz",
        )

        with mock.patch(
            "motus.utils.scheduler.urllib.request.urlopen", return_value=fake_resp
        ) as m:
            result = await s.schedule(
                "daily-digest",
                schedule_type="rate",
                expression="rate(5 minutes)",
                message="Time to check.",
            )

        self.assertEqual(result, "notif_001")
        req = m.call_args[0][0]
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.full_url, "https://api.example.com/notifications")
        self.assertEqual(req.get_header("Authorization"), "Bearer key_123")

        body = json.loads(req.data)
        self.assertEqual(body["project_id"], "proj_xyz")
        self.assertEqual(body["session_id"], "sess_abc")
        self.assertEqual(body["session_token"], "ses_tok")
        self.assertEqual(body["name"], "daily-digest")
        self.assertEqual(body["schedule_type"], "rate")
        self.assertEqual(body["schedule_expression"], "rate(5 minutes)")
        self.assertEqual(body["message"], "Time to check.")
        self.assertEqual(body["timezone"], "UTC")

    async def test_custom_timezone(self):
        fake_resp = _mock_response({"notification_id": "notif_002"})
        s = CloudScheduler(
            api_url="https://api.example.com",
            api_key="k",
            session_id="s",
            session_token="t",
            project_id="p",
        )

        with mock.patch(
            "motus.utils.scheduler.urllib.request.urlopen", return_value=fake_resp
        ) as m:
            await s.schedule(
                "tz-test",
                schedule_type="cron",
                expression="cron(0 9 * * ? *)",
                message="morning",
                timezone="America/New_York",
            )

        body = json.loads(m.call_args[0][0].data)
        self.assertEqual(body["timezone"], "America/New_York")


class TestCloudSchedulerRemove(unittest.IsolatedAsyncioTestCase):
    async def test_sends_delete(self):
        fake_resp = _mock_response()
        s = CloudScheduler(
            api_url="https://api.example.com",
            api_key="key_123",
            session_id="s",
            session_token="t",
            project_id="p",
        )

        with mock.patch(
            "motus.utils.scheduler.urllib.request.urlopen", return_value=fake_resp
        ) as m:
            await s.remove("notif_001")

        req = m.call_args[0][0]
        self.assertEqual(req.get_method(), "DELETE")
        self.assertEqual(
            req.full_url, "https://api.example.com/notifications/notif_001"
        )


class TestCloudSchedulerList(unittest.IsolatedAsyncioTestCase):
    async def test_sends_get(self):
        fake_resp = _mock_response(
            [{"notification_id": "n1"}, {"notification_id": "n2"}]
        )
        s = CloudScheduler(
            api_url="https://api.example.com",
            api_key="k",
            session_id="s",
            session_token="t",
            project_id="p",
        )

        with mock.patch(
            "motus.utils.scheduler.urllib.request.urlopen", return_value=fake_resp
        ) as m:
            result = await s.list()

        self.assertEqual(len(result), 2)
        self.assertEqual(m.call_args[0][0].get_method(), "GET")


class TestCloudSchedulerErrors(unittest.IsolatedAsyncioTestCase):
    async def test_http_error(self):
        import urllib.error

        error = urllib.error.HTTPError(
            url="https://api.example.com/notifications",
            code=422,
            msg="Unprocessable",
            hdrs={},
            fp=BytesIO(b'{"detail": "bad"}'),
        )
        s = CloudScheduler(
            api_url="https://api.example.com",
            api_key="k",
            session_id="s",
            session_token="t",
            project_id="p",
        )

        with mock.patch(
            "motus.utils.scheduler.urllib.request.urlopen", side_effect=error
        ):
            with self.assertRaises(SchedulerError) as ctx:
                await s.schedule(
                    "fail",
                    schedule_type="rate",
                    expression="rate(1 hour)",
                    message="msg",
                )
        self.assertEqual(ctx.exception.status, 422)

    async def test_url_error(self):
        import urllib.error

        error = urllib.error.URLError("Connection refused")
        s = CloudScheduler(
            api_url="https://api.example.com",
            api_key="k",
            session_id="s",
            session_token="t",
            project_id="p",
        )

        with mock.patch(
            "motus.utils.scheduler.urllib.request.urlopen", side_effect=error
        ):
            with self.assertRaises(SchedulerError):
                await s.list()


# ======================================================================
# LocalScheduler
# ======================================================================


class TestLocalScheduler(unittest.IsolatedAsyncioTestCase):
    async def test_schedule_and_dispatch(self):
        scheduler = LocalScheduler()
        calls = []
        scheduler.on("test-job", lambda: calls.append("fired"))

        job_id = await scheduler.schedule(
            "test-job", schedule_type="rate", expression="rate(1 seconds)"
        )

        self.assertIn(job_id, scheduler._jobs)

        # Manually fire the scheduler (simulates the run loop advancing)
        scheduler._scheduler.run(blocking=False)
        # Job is scheduled 1s in the future, so nothing fires yet
        self.assertEqual(len(calls), 0)

    async def test_remove_cancels_job(self):
        scheduler = LocalScheduler()
        scheduler.on("test-job", lambda: None)
        job_id = await scheduler.schedule(
            "test-job", schedule_type="rate", expression="rate(60 seconds)"
        )
        await scheduler.remove(job_id)
        self.assertNotIn(job_id, scheduler._jobs)

    async def test_on_registers_handler(self):
        scheduler = LocalScheduler()

        def handler():
            pass

        scheduler.on("my-job", handler)
        self.assertIs(scheduler._handlers["my-job"], handler)

    async def test_async_handler(self):
        scheduler = LocalScheduler()
        calls = []

        async def async_handler():
            calls.append("async-fired")

        scheduler.on("async-job", async_handler)
        # Directly test the _fire path
        job = scheduler._jobs.get("nonexistent")  # won't exist yet
        # Schedule and then simulate fire
        job_id = await scheduler.schedule(
            "async-job", schedule_type="at", expression="at(2025-01-01T00:00:00)"
        )
        job = scheduler._jobs[job_id]
        scheduler._fire(job)
        await asyncio.sleep(0)  # let the event loop process the task
        self.assertEqual(calls, ["async-fired"])

    async def test_one_shot_removes_after_fire(self):
        scheduler = LocalScheduler()
        scheduler.on("once", lambda: None)
        job_id = await scheduler.schedule(
            "once", schedule_type="at", expression="at(2025-01-01T00:00:00)"
        )
        job = scheduler._jobs[job_id]
        scheduler._fire(job)
        self.assertNotIn(job_id, scheduler._jobs)

    async def test_recurring_reschedules_after_fire(self):
        scheduler = LocalScheduler()
        scheduler.on("recurring", lambda: None)
        job_id = await scheduler.schedule(
            "recurring", schedule_type="rate", expression="rate(30 seconds)"
        )
        job = scheduler._jobs[job_id]
        queue_len_before = len(scheduler._scheduler.queue)
        scheduler._fire(job)
        # Job should still exist and be rescheduled
        self.assertIn(job_id, scheduler._jobs)
        self.assertGreater(len(scheduler._scheduler.queue), queue_len_before)


# ======================================================================
# _parse_interval
# ======================================================================


class TestParseInterval(unittest.TestCase):
    def test_rate_minutes(self):
        self.assertEqual(_parse_interval("rate", "rate(5 minutes)"), 300)

    def test_rate_hours(self):
        self.assertEqual(_parse_interval("rate", "rate(2 hours)"), 7200)

    def test_rate_seconds(self):
        self.assertEqual(_parse_interval("rate", "rate(30 seconds)"), 30)

    def test_rate_singular(self):
        self.assertEqual(_parse_interval("rate", "rate(1 minute)"), 60)

    def test_at_returns_1(self):
        self.assertEqual(_parse_interval("at", "at(2026-04-15T09:00:00)"), 1)

    def test_cron_returns_60(self):
        self.assertEqual(_parse_interval("cron", "cron(0 9 * * ? *)"), 60)
