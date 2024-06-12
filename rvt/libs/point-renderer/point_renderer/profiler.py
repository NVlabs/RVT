# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import time
import torch
import glob
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def colorize_time(elapsed):
    if elapsed > 1e-3:
        return bcolors.FAIL + "{:.3e}".format(elapsed) + bcolors.ENDC
    elif elapsed > 1e-4:
        return bcolors.WARNING + "{:.3e}".format(elapsed) + bcolors.ENDC
    elif elapsed > 1e-5:
        return bcolors.OKBLUE + "{:.3e}".format(elapsed) + bcolors.ENDC
    else:
        return "{:.3e}".format(elapsed)

def print_gpu_memory():
    torch.cuda.empty_cache()
    print(f"{torch.cuda.memory_allocated()//(1024*1024)} mb")


class PerfTimer():
    def __init__(self, activate=False, show_memory=False, print_mode=True):
        self.activate = activate
        if activate:
            self.show_memory = show_memory
            self.print_mode = print_mode
            self.init()

    def init(self):
        self.reset()
        self.loop_totals_cpu = {}
        self.loop_totals_gpu = {}
        self.loop_counts = {}


    def reset(self):
        if self.activate:
            self.counter = 0
            self.prev_time = time.perf_counter()
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.prev_time_gpu = self.start.record()

    def check(self, name=None):
        if self.activate:
            cpu_time = time.perf_counter() - self.prev_time
          
            self.end.record()
            torch.cuda.synchronize()

            gpu_time = self.start.elapsed_time(self.end) / 1e3

            # Keep track of averages. For this to work, keys need to be unique in a global scope
            if name not in self.loop_counts:
                self.loop_totals_cpu[name] = 0
                self.loop_totals_gpu[name] = 0
                self.loop_counts[name] = 0
            self.loop_totals_gpu[name] += gpu_time
            self.loop_totals_cpu[name] += cpu_time
            self.loop_counts[name] += 1

            if self.print_mode and name:
                cpu_time_disp = colorize_time(cpu_time)
                gpu_time_disp = colorize_time(gpu_time)
                cpu_time_disp_avg = colorize_time(self.loop_totals_cpu[name] / self.loop_counts[name])
                gpu_time_disp_avg = colorize_time(self.loop_totals_gpu[name] / self.loop_counts[name])

                if name:
                    print(f"CPU Checkpoint {name}: {cpu_time_disp}s (Avg: {cpu_time_disp_avg}s)")
                    print(f"GPU Checkpoint {name}: {gpu_time_disp}s (Avg: {gpu_time_disp_avg}s)")
                else:
                    print("CPU Checkpoint {}: {} s".format(self.counter, cpu_time_disp))
                    print("GPU Checkpoint {}: {} s".format(self.counter, gpu_time_disp))
                if self.show_memory:
                    #torch.cuda.empty_cache()
                    print(f"{torch.cuda.memory_allocated()//1048576}MB")
                

            self.prev_time = time.perf_counter()
            self.prev_time_gpu = self.start.record()
            self.counter += 1
            return cpu_time, gpu_time
    
    def get_avg_cpu_time(self, name):
        return self.loop_totals_cpu[name] / self.loop_counts[name]

    def get_avg_gpu_time(self, name):
        return self.loop_totals_gpu[name] / self.loop_counts[name]
