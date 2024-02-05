import logging
import time
import functools

import torch
import psutil
import os


def get_gpu_memory(device_index=None):
    return torch.cuda.memory_allocated(device_index)


def get_gpu_max_memory(device_index=None):
    return torch.cuda.max_memory_allocated(device_index)


def get_device_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def get_cpu_swap_memory():
    return psutil.swap_memory().used


class FunctionTimer:
    def __init__(self, host_timer=False):
        self.host_timer = host_timer
        self.log = {}

    def __call__(self, func):
        def _host_timer_wrapper(*args, **kwargs):
            before_memory = get_device_memory()
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            after_memory = get_device_memory()
            self.log[func.__name__] = {
                "timer": end_time - start_time,
                "cpu_memory": (before_memory - after_memory) >> 20,
            }
            return result

        def _device_timer_wrapper(*args, **kwargs):
            before_memory = get_gpu_memory()
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = func(*args, **kwargs)
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            after_memory = get_gpu_memory()
            torch.cuda.current_stream().wait_event(end_event)
            torch.cuda.synchronize()
            self.log[func.__name__] = {
                "timer": start_event.elapsed_time(end_event),
                "gpu_memory": (before_memory - after_memory) >> 20,
            }
            return result

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.host_timer:
                return _host_timer_wrapper(*args, **kwargs)
            return _device_timer_wrapper(*args, **kwargs)

    def get_log(self):
        return self.log


class Timer:
    def __init__(self):
        self.timers = {
            "cpu": {},
            "gpu": {},
        }
        self.timer_info = {}  # synchronized.
        self.is_synchromized = False

    def start_timer(self, name, host_timer=False):
        if host_timer:
            if name in self.timers["cpu"]:
                logging.warning(f"timer for {name} already exist")
                return
            start_time = time.time()
            self.timers["cpu"][name] = [start_time]
        else:
            if name in self.timers["gpu"]:
                logging.warning(f"timer for {name} already exist")
                return
            self.is_synchromized = False
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            self.timers["gpu"][name] = [start_event]

    def stop_timer(self, name):
        if name in self.timers["cpu"]:
            end_time = time.time()
            self.timers["cpu"][name].append(end_time)
        if name in self.timers["gpu"]:
            self.is_synchromized = False
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            self.timers["gpu"][name].append(end_event)

    def _calculate_elapse_time(self):
        for name, timer in self.timers["cpu"].items():
            assert len(timer) == 2
            self.timer_info[name] = (timer[1] - timer[0]) * 1000
        if not self.is_synchromized:
            for name, events in self.timers["gpu"].items():
                assert len(events) == 2
                torch.cuda.current_stream().wait_event(events[1])
                torch.cuda.synchronize()
                self.timer_info[name] = events[0].elapsed_time(events[1])
            self.is_synchromized = True

    def get_info(self):
        if not self.is_synchromized:
            self._calculate_elapse_time()
        return self.timer_info
