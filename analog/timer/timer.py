import logging
import time
import functools

import torch


def get_gpu_memory(device_index=None):
    return torch.cuda.memory_allocated(device_index)


def get_gpu_max_memory(device_index=None):
    return torch.cuda.max_memory_allocated(device_index)


class FunctionTimer:
    log = {}

    @classmethod
    def _wrap_function(cls, func, label, host_timer):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if host_timer:
                return cls._host_timer_wrapper(func, label, *args, **kwargs)
            else:
                return cls._device_timer_wrapper(func, label, *args, **kwargs)

        return wrapper

    @classmethod
    def _host_timer_wrapper(cls, func, label, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if label not in cls.log:
            cls.log[label] = [
                {
                    "time_delta": end_time - start_time,
                }
            ]
        else:
            cls.log[label].append(
                {
                    "time_delta": end_time - start_time,
                }
            )
        return result

    @classmethod
    def _device_timer_wrapper(cls, func, label, *args, **kwargs):
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = func(*args, **kwargs)
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        torch.cuda.current_stream().wait_event(end_event)
        torch.cuda.synchronize()
        if label not in cls.log:
            cls.log[label] = [
                {
                    "time_delta": start_event.elapsed_time(end_event)
                    / 1000,  # turn to seconds
                }
            ]
        else:
            cls.log[label].append(
                {
                    "time_delta": start_event.elapsed_time(end_event)
                    / 1000,  # turn to seconds
                }
            )
        return result

    @classmethod
    def timer(cls, label_or_func=None):
        host_timer = getattr(
            cls, "host_timer", False
        )  # Fallback to False if not defined

        def decorator(func):
            label = label_or_func if isinstance(label_or_func, str) else func.__name__
            return cls._wrap_function(func, label, host_timer)

        if callable(label_or_func):
            return decorator(label_or_func)
        return decorator

    @classmethod
    def get_log(cls):
        return cls.log

    @classmethod
    def print_log(cls):
        print(
            "###########################################################################"
        )
        print(
            "################################ TIMER LOG ################################"
        )
        header = f"{'Label':<50} | {'Total Time (sec)':>20}"
        print(header)
        print("-" * len(header))
        for label, details in cls.log.items():
            sum_time = 0
            for log_entry in details:
                time_delta = log_entry.get("time_delta", 0)
                sum_time += time_delta
            # truncate 47 letters if the label is longer than 50.
            display_label = (label[:47] + "...") if len(label) > 50 else label
            row = f"{display_label:<50} | {sum_time:>20.4f}"
            print(row)


class HostFunctionTimer(FunctionTimer):
    host_timer = True


class DeviceFunctionTimer(FunctionTimer):
    if torch.cuda.is_available():
        host_timer = False
    else:
        logging.warning("CUDA is not set, setting the timer is set to host timer.")
        host_timer = True


class Timer:
    def __init__(self):
        self.timers = {
            "cpu": {},
            "gpu": {},
        }
        self.timer_info = {}  # synchronized.
        self.is_synchronized = False

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
            self.is_synchronized = False
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            self.timers["gpu"][name] = [start_event]

    def stop_timer(self, name):
        if name in self.timers["cpu"]:
            end_time = time.time()
            self.timers["cpu"][name].append(end_time)
        if name in self.timers["gpu"]:
            self.is_synchronized = False
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            self.timers["gpu"][name].append(end_event)

    def _calculate_elapse_time(self):
        for name, timer in self.timers["cpu"].items():
            assert len(timer) == 2
            self.timer_info[name] = (timer[1] - timer[0]) * 1000
        if not self.is_synchronized:
            for name, events in self.timers["gpu"].items():
                assert len(events) == 2
                torch.cuda.current_stream().wait_event(events[1])
                torch.cuda.synchronize()
                self.timer_info[name] = events[0].elapsed_time(events[1])
            self.is_synchronized = True

    def get_info(self):
        if not self.is_synchronized:
            self._calculate_elapse_time()
        return self.timer_info
