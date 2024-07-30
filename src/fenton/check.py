# %%
import nvsmi
import time
import logging

# %%
list(nvsmi.get_gpus())
# %%
check_every_n_seconds = 1
min_util = 5
check_seconds = 60*30

# %%
# make a queue of the last 10 values
# if the average of the last 10 values is below 20% for 10 seconds, then power off the gpu

import collections
qs = [
    collections.deque(maxlen=check_seconds//check_every_n_seconds) for _ in list(nvsmi.get_gpus())
]
while True:
    gpus = nvsmi.get_gpus()
    for gpu, q in zip(gpus, qs):
        q.append(gpu.gpu_util)
        if len(q) == check_seconds//check_every_n_seconds and max(q) < min_util:
            print("Powering off GPU")
            break
    logging.warning(f"GPU Utilization: {gpu.gpu_util}")
    time.sleep(check_every_n_seconds)
    
# for i in range(10):
#     gpus = nvsmi.get_gpus()
#     for gpu in gpus:
#         util.append(gpu.gpu_util)
#     time.sleep(check_every_n_seconds)
# %%
