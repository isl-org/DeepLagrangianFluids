import os
import time
import resource


class IterationTimer:

    def __init__(self):
        self.start_iteration = None
        self.start_time = None

    def get_avg_iteration_time(self, iteration):
        """Returns the averaged time per iteration since the last call

        iteration: int
            The current iteration

        Returns the average time per iteration or None
        """
        if not self.start_iteration or self.start_iteration >= iteration:
            self.start_iteration = iteration
            self.start_time = time.time()
            return None
        else:
            now = time.time()
            avg_iteration_time = (now - self.start_time) / (
                iteration - self.start_iteration)
            self.start_iteration = iteration
            self.start_time = now
            return avg_iteration_time


class CPULoad:

    def __init__(self):
        self.start_cpu_user_time = None
        self.start_cpu_sys_time = None
        self.start_wall_time = None

    def get_avg_cpu_load(self):
        """Returns the average cpu load since the last call
        
        Returns the user and system time fraction per second as tuple or None
        """
        if not self.start_wall_time:
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            self.start_wall_time = time.time()
            self.start_cpu_user_time = rusage.ru_utime
            self.start_cpu_sys_time = rusage.ru_stime
            return None
        else:
            now = time.time()
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            time_delta = now - self.start_wall_time
            avg_user_time = (rusage.ru_utime -
                             self.start_cpu_user_time) / time_delta
            avg_sys_time = (rusage.ru_stime -
                            self.start_cpu_sys_time) / time_delta
            self.start_wall_time = now
            self.start_cpu_user_time = rusage.ru_utime
            self.start_cpu_sys_time = rusage.ru_stime
            return avg_user_time, avg_sys_time


class GPUAccounting:
    _initialized = False
    device_handles = {}
    device_handles_with_accounting = {}
    pid = os.getpid()

    def __init__(self):
        from .nvml import HAVE_NVML, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetAccountingMode, NVML_FEATURE_ENABLED
        if not GPUAccounting._initialized and HAVE_NVML:
            dev_count = nvmlDeviceGetCount()
            for idx in range(dev_count):
                device = nvmlDeviceGetHandleByIndex(idx)
                GPUAccounting.device_handles[idx] = device
                if nvmlDeviceGetAccountingMode(device) == NVML_FEATURE_ENABLED:
                    GPUAccounting.device_handles_with_accounting[idx] = device
            GPUAccounting._initialized = True

    def get_accounting_stats(self):
        """Returns the accounting stats for all gpus for the pid
        Returns an empty dict if there is no gpu or accounting is disabled for all gpus.
        """
        from .nvml import nvmlDeviceGetAccountingStats
        result = {}
        for idx, device in GPUAccounting.device_handles_with_accounting.items():
            stats = nvmlDeviceGetAccountingStats(device, GPUAccounting.pid)
            if not stats is None:
                result[idx] = stats
        return result
