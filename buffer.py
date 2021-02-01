import torch as t
import numpy as np

class AbstractBuffer:
    """
    External interface:
    self.filled_buffer
        the number of (s, a, sp, r, d) tuples in the buffer.
    add_state(s, a, sp, r, d) 
        (where d is the final state)
    random_batch(T)
        (a non-contiguous random batch of size min(T, self.filled_buffer))
    contiguous_batch(T)
        (a contiguous random batch of size min(T, self.filled_buffer))
    """
    def __init__(self, state_shape, storage_device='cpu', output_device='cpu', output_dtype=t.float32, init_buffer=32):
        self.storage_device = storage_device
        self.output_device = output_device
        self.output_dtype = output_dtype
        self
        self.filled_buffer = 0
        self.ratio = 4
        self.buffer_size = init_buffer
        self.r  = t.zeros((init_buffer, 1),            device=storage_device)
        self.s  = t.zeros((init_buffer, *state_shape), device=storage_device)
        self.sp = t.zeros((init_buffer, *state_shape), device=storage_device)
        self.d  = t.zeros((init_buffer, 1),            device=storage_device)

    def expand_buffer(self, old_buf):
        assert old_buf.shape[0] == self.filled_buffer
        new_buf = t.zeros((self.buffer_size, *old_buf.shape[1:]), 
                          dtype=old_buf.dtype, 
                          device=self.storage_device)
        new_buf[:self.filled_buffer] = old_buf
        return new_buf

    def add_state(self, s, a, sp, r, d):
        if self.filled_buffer == self.buffer_size:
            self.buffer_size = self.ratio*self.buffer_size
            self.s  = self.expand_buffer(self.s)
            self.a  = self.expand_buffer(self.a)
            self.sp = self.expand_buffer(self.sp)
            self.r  = self.expand_buffer(self.r)
            self.d  = self.expand_buffer(self.d)

        i = self.filled_buffer 
        self.s[i, :]  = t.tensor(s,  device=self.storage_device)
        if isinstance(a, int):
            self.a[i, 0]  = a
        else:
            self.a[i, :]  = t.tensor(a,  device=self.storage_device)
        self.sp[i, :] = t.tensor(sp, device=self.storage_device)

        self.r[i, 0]  = r
        self.d[i, 0]  = d
        self.filled_buffer += 1

    def random_idxs(self, T):
        return  t.randperm(self.filled_buffer, device=self.storage_device)[:min(T, self.filled_buffer)]

    def contiguous_idxs(self, T):
        if self.filled_buffer < T:
            return range(self.filled_buffer)
        else:
            start = np.random.randint(self.filled_buffer)
            if start + T < self.filled_buffer:
                return range(start, start+T)
            else:
                return list(range(start, self.filled_buffer)) + list(range(T-(self.filled_buffer-start)))

    def batch(self, idxs):
        kwargs = {'device': self.output_device, 'dtype': self.output_dtype}
        return (self.s[idxs].to(**kwargs),
                self.a[idxs].to(**kwargs),
                self.sp[idxs].to(**kwargs),
                self.r[idxs].to(**kwargs),
                self.d[idxs].to(**kwargs))

    def random_batch(self, T):
        return self.batch(self.random_idxs(T))

    def contiguous_batch(self, T):
        return self.batch(self.contiguous_idxs(T))

        
        

class BufferContinuous(AbstractBuffer):
    def __init__(self, state_shape, action_features, storage_device='cpu', output_device='cpu', output_dtype=t.float32, init_buffer=32):
        super().__init__(state_shape, storage_device, output_device, output_dtype, init_buffer)
        self.a = t.zeros(init_buffer, action_features, device=storage_device)

class BufferDiscrete(AbstractBuffer):
    def __init__(self, state_shape, storage_device='cpu', output_device='cpu', output_dtype=t.float32,  init_buffer=32):
        super().__init__(state_shape, storage_device, output_device, output_dtype, init_buffer)
        self.a = t.zeros(init_buffer, 1, dtype=t.int, device=storage_device)

if __name__ == "__main__":
    buff = BufferDiscrete(state_shape=(2,), init_buffer=2)
    buff.add_state(t.ones(2), 2, t.ones(2), 3., 0)
    buff.add_state(t.ones(2), 2, t.ones(2), 3., 0)
    buff.add_state(t.ones(2), 2, t.ones(2), 3., 0)
    buff.add_state(t.ones(2), 2, t.ones(2), 3., 0)

    s, a, sp, r, d = buff.random_batch(2)
    assert s.shape[0] == 2
    s, a, sp, r, d = buff.random_batch(4)
    assert s.shape[0] == 4
    s, a, sp, r, d = buff.random_batch(6)
    assert s.shape[0] == 4

    s, a, sp, r, d = buff.contiguous_batch(3)
    assert s.shape[0] == 3
    s, a, sp, r, d = buff.contiguous_batch(3)
    assert s.shape[0] == 3
    s, a, sp, r, d = buff.contiguous_batch(4)
    assert s.shape[0] == 4
    s, a, sp, r, d = buff.contiguous_batch(6)
    assert s.shape[0] == 4
