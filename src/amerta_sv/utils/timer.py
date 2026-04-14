import time
from contextlib import contextmanager
class Timer:
    def __init__(self):
        self.times={}; self.start_times={}
    def start(self,n): self.start_times[n]=time.time()
    def stop(self,n):
        if n in self.start_times:
            e=time.time()-self.start_times.pop(n); self.times[n]=e; return e
        return 0.0
    @contextmanager
    def time_section(self,n):
        self.start(n); yield; self.stop(n)
    def get_times(self): return dict(self.times)
    def total(self): return sum(self.times.values())
