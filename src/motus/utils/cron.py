import asyncio
import datetime
import sched
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable


class Cron:
    @dataclass(frozen=True)  # For __hash__ and __eq__
    class Job:
        minute: tuple[int, ...] | None
        hour: tuple[int, ...] | None
        day_of_month: tuple[int, ...] | None
        month: tuple[int, ...] | None
        day_of_week: tuple[int, ...] | None
        absolute: int | None
        interval: int | None
        func: Callable[[], None]

    class Schedule:
        minute: deque[int] | None
        hour: deque[int] | None
        day_of_month: deque[int] | None
        month: deque[int] | None
        day_of_week: deque[int] | None
        absolute: int | None
        interval: int | None

        def __init__(self, job: "Cron.Job") -> None:
            self.minute = deque(job.minute) if job.minute is not None else None
            self.hour = deque(job.hour) if job.hour is not None else None
            self.day_of_month = (
                deque(job.day_of_month) if job.day_of_month is not None else None
            )
            self.month = deque(job.month) if job.month is not None else None
            self.day_of_week = (
                deque(job.day_of_week) if job.day_of_week is not None else None
            )
            self.absolute = job.absolute
            self.interval = job.interval

        def __next__(self) -> datetime.datetime:
            if self.absolute is not None:  # Absolute time
                dt = datetime.datetime.fromtimestamp(self.absolute)
                if dt < datetime.datetime.now():
                    raise StopIteration
            elif self.interval is not None:  # Interval in seconds
                dt = datetime.datetime.now() + datetime.timedelta(seconds=self.interval)
            else:  # Cron expression
                now = datetime.datetime.now().replace(second=0, microsecond=0)
                dt = now + datetime.timedelta(minutes=1)
                while self.month and dt.month not in self.month:
                    dt += datetime.timedelta(days=1)
                    dt = dt.replace(day=1, hour=0, minute=0)
                while self.day_of_month and dt.day not in self.day_of_month:
                    dt += datetime.timedelta(days=1)
                    dt = dt.replace(hour=0, minute=0)
                while self.day_of_week and dt.weekday() not in self.day_of_week:
                    dt += datetime.timedelta(days=1)
                    dt = dt.replace(hour=0, minute=0)
                while self.hour and dt.hour not in self.hour:
                    dt += datetime.timedelta(hours=1)
                    dt = dt.replace(minute=0)
                while self.minute and dt.minute not in self.minute:
                    dt += datetime.timedelta(minutes=1)

            return dt

    def __init__(self) -> None:
        self.scheduler = sched.scheduler(time.time, lambda _: None)
        self.jobs: dict[Cron.Job, Cron.Schedule] = {}

    def create_cron(
        self,
        minute: int | Iterable[int] | None,
        hour: int | Iterable[int] | None,
        day_of_month: int | Iterable[int] | None,
        month: int | Iterable[int] | None,
        day_of_week: int | Iterable[int] | None,
        func: Callable[[], None],
    ) -> Job:
        # Is there a cleaner way to do this?
        def tupler(x: int | Iterable[int] | None) -> tuple[int, ...]:
            return (
                ()
                if x is None
                else tuple(sorted(x) if isinstance(x, Iterable) else [x])
            )

        job = Cron.Job(
            tupler(minute),
            tupler(hour),
            tupler(day_of_month),
            tupler(month),
            tupler(day_of_week),
            None,
            None,
            func,
        )

        self.schedule_job(job)

        return job

    def create_absolute(self, absolute: int, func: Callable[[], None]):
        job = Cron.Job(None, None, None, None, None, absolute, None, func)

        self.schedule_job(job)

        return job

    def create_interval(self, interval: int, func: Callable[[], None]):
        job = Cron.Job(None, None, None, None, None, None, interval, func)

        self.schedule_job(job)

        return job

    def schedule_job(self, job: "Cron.Job"):
        self.jobs[job] = Cron.Schedule(job)

        def schedule(run: bool = True):
            if job in self.jobs:
                try:
                    self.scheduler.enterabs(
                        next(self.jobs[job]).timestamp(),
                        0,
                        schedule,
                    )
                except StopIteration:
                    pass

                if run:
                    job.func()

        schedule(False)

    def remove(self, job: Job) -> None:
        del self.jobs[job]

    async def run(self) -> None:
        while True:
            self.scheduler.run(blocking=False)
            await asyncio.sleep(1)
