from py3nvml.py3nvml import *
"""
"""
def gpu_energy():
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    power_reading = []
    for i in range(0, deviceCount):
        try:
            handle = nvmlDeviceGetHandleByIndex(i)
            pow_draw = (nvmlDeviceGetPowerUsage(handle) / 1000.0)
            power_reading.append(pow_draw)
        except NVMLError as e:
            pass

    if len(power_reading) > 0:
       energy = [pow_draw for pow_draw in power_reading]
      
    return energy[0]



