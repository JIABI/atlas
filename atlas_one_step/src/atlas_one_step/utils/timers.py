import time
class Timer:
    def __enter__(self): self.t=time.time(); return self
    def __exit__(self,*a): self.elapsed=time.time()-self.t
