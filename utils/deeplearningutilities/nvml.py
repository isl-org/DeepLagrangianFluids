import sys
from ctypes import *
from ctypes.util import find_library
import atexit

NVML_SUCCESS = 0
NVML_FEATURE_DISABLED = 0
NVML_FEATURE_ENABLED = 1


class nvmlUtilization_t(Structure):
    _fields_ = [('gpu', c_uint), ('memory', c_uint)]


class nvmlProcessInfo_t(Structure):
    _fields_ = [('pid', c_uint), ('usedGpuMemory', c_ulonglong)]


class nvmlAccountingStats_t(Structure):
    _fields_ = [
        ('gpuUtilization', c_uint),
        ('memoryUtilization', c_uint),
        ('maxMemoryUsage', c_ulonglong),
        ('time', c_ulonglong),
        ('startTime', c_ulonglong),
        ('isRunning', c_uint),
        ('reserved', c_uint * 5),
    ]


def nvmlInit():
    status = nvml.nvmlInit()
    if status != NVML_SUCCESS:
        raise RuntimeError('nvmlInit failed')


def nvmlShutdown():
    status = nvml.nvmlShutdown()
    if status != NVML_SUCCESS:
        raise RuntimeError('nvmlShutdown failed')


def nvmlDeviceGetCount():
    deviceCount = c_uint()
    status = nvml.nvmlDeviceGetCount(pointer(deviceCount))
    if status != NVML_SUCCESS:
        return None
    return deviceCount.value


def nvmlDeviceGetHandleByIndex(index):
    idx = c_uint(index)
    device = c_void_p()
    status = nvml.nvmlDeviceGetHandleByIndex(idx, pointer(device))
    if status != NVML_SUCCESS:
        return None
    return device


def nvmlDeviceGetName(device):
    name = create_string_buffer(256)
    status = nvml.nvmlDeviceGetName(device, name, 256)
    if status != NVML_SUCCESS:
        return None
    return name.value.decode('utf-8')


def nvmlDeviceGetUtilizationRates(device):
    utilization = nvmlUtilization_t()
    status = nvml.nvmlDeviceGetUtilizationRates(device, pointer(utilization))
    if status != NVML_SUCCESS:
        return None
    return utilization.gpu, utilization.memory


def nvmlDeviceGetComputeRunningProcesses(device):
    nvmlProcessInfo_t_Array10 = nvmlProcessInfo_t * 10
    infos = nvmlProcessInfo_t_Array10()
    infoCount = c_uint(10)
    status = nvml.nvmlDeviceGetComputeRunningProcesses(device,
                                                       pointer(infoCount),
                                                       pointer(infos))
    if status != NVML_SUCCESS:
        return None
    result = []
    for i in range(infoCount.value):
        result.append({
            'pid': infos[i].pid,
            'usedGpuMemory': infos[i].usedGpuMemory
        })
    return result


def nvmlDeviceGetAccountingBufferSize(device):
    bufferSize = c_uint()
    status = nvml.nvmlDeviceGetAccountingBufferSize(device, pointer(bufferSize))
    if status != NVML_SUCCESS:
        return None
    return bufferSize.value


def nvmlDeviceGetAccountingMode(device):
    mode = c_int()
    status = nvml.nvmlDeviceGetAccountingMode(device, pointer(mode))
    if status != NVML_SUCCESS:
        return None
    return mode.value


def nvmlDeviceGetAccountingPids(device):
    pids_Array = c_uint * 1024
    pids = pids_Array()
    count = c_uint(1024)
    status = nvml.nvmlDeviceGetAccountingPids(device, pointer(count),
                                              pointer(pids))
    if status != NVML_SUCCESS:
        return None
    result = []
    for i in range(count.value):
        result.append(pids[i])
    return result


def nvmlDeviceGetAccountingStats(device, pid):
    cpid = c_uint(pid)
    stats = nvmlAccountingStats_t()
    status = nvml.nvmlDeviceGetAccountingStats(device, cpid, pointer(stats))
    if status != NVML_SUCCESS:
        return None
    return {
        'gpuUtilization': stats.gpuUtilization,
        'isRunning': stats.isRunning,
        'maxMemoryUsage': stats.maxMemoryUsage,
        'memoryUtilization': stats.memoryUtilization,
        #'reserved': stats.reserved,
        'startTime': stats.startTime,
        'time': stats.time
    }


def getProcessName(pid):
    try:
        with open('/proc/{0}/comm'.format(pid), 'r') as f:
            name = f.read().strip()
    except:
        return None
    return name


if not find_library('nvidia-ml') is None:

    nvml = CDLL('libnvidia-ml.so')
    atexit.register(nvmlShutdown)
    nvmlInit()
    HAVE_NVML = True
else:
    HAVE_NVML = False
