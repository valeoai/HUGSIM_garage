"""The promises agents/client.py makes to closed_loop.py, checked over real FIFOs.

Standard library and numpy only, no GPU and no scene data:

    python agents/tools/test_client.py        # or: pytest agents/tools/test_client.py

Each test plays the simulator's side of the protocol against ``serve`` running in a thread, and
fails rather than hangs: every pipe operation of the simulator's side has a deadline.
"""

import os
import pickle
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents.client import serve  # noqa: E402

DEADLINE = 10.0  # s


def _open(path, flags):
    """open() a FIFO without blocking past DEADLINE (the other side may never come)."""
    end = time.monotonic() + DEADLINE
    while True:
        try:
            return os.open(path, flags | os.O_NONBLOCK)
        except OSError:  # ENXIO: no reader yet
            if time.monotonic() > end:
                raise TimeoutError(f'nobody opened {path}')
            time.sleep(0.01)


class Simulator:
    """The simulator's side: send a payload, read the reply, as closed_loop.py's FifoAgent does."""

    def __init__(self, step):
        self.dir = tempfile.mkdtemp()
        self.result = {}
        self.thread = threading.Thread(
            target=lambda: self.result.setdefault('planned', serve(self.dir, step, name='test')),
            daemon=True)
        self.thread.start()
        self.obs_pipe = os.path.join(self.dir, 'obs_pipe')
        self.plan_pipe = os.path.join(self.dir, 'plan_pipe')
        end = time.monotonic() + DEADLINE
        while not os.path.exists(self.plan_pipe):
            assert time.monotonic() < end, 'serve made no pipes'
            time.sleep(0.01)

    def send(self, payload):
        fd = _open(self.obs_pipe, os.O_WRONLY)
        os.set_blocking(fd, True)
        with os.fdopen(fd, 'wb') as pipe:
            pipe.write(payload)

    def ask(self, payload):
        self.send(payload)
        # The agent opens plan_pipe for writing once it has an answer; a blocking open here
        # would hang the test if it never does, so wait for it by polling a reader.
        fd = os.open(self.plan_pipe, os.O_RDONLY | os.O_NONBLOCK)
        os.set_blocking(fd, True)
        end = time.monotonic() + DEADLINE
        data = b''
        while True:
            chunk = os.read(fd, 1 << 16)
            if chunk:
                data += chunk
            elif data:
                break
            elif time.monotonic() > end:
                os.close(fd)
                raise TimeoutError('no answer')
            else:
                time.sleep(0.01)
        os.close(fd)
        return pickle.loads(data)

    def finish(self):
        self.thread.join(DEADLINE)
        assert not self.thread.is_alive(), 'serve did not return'
        return self.result['planned']


def plan(obs, info):
    return np.zeros((6, 2), dtype=np.float32)


def test_answers_each_request_and_returns_on_done():
    sim = Simulator(plan)
    assert sim.ask(pickle.dumps(({}, {}))).shape == (6, 2)
    assert sim.ask(pickle.dumps(({}, {}))).shape == (6, 2)
    sim.send(pickle.dumps('Done'))
    assert sim.finish() == 2


def test_a_step_that_raises_is_answered_none_then_done_is_read():
    def broken(obs, info):
        raise ValueError('model failed')
    sim = Simulator(broken)
    assert sim.ask(pickle.dumps(({}, {}))) is None
    assert 'model failed' in open(os.path.join(sim.dir, 'agent_crash.txt')).read()
    sim.send(pickle.dumps('Done'))
    assert sim.finish() == 0


def test_an_unreadable_request_is_answered_none():
    sim = Simulator(plan)
    assert sim.ask(b'not a pickle') is None
    assert sim.ask(pickle.dumps(({}, {}))).shape == (6, 2)
    sim.send(pickle.dumps('Done'))
    assert sim.finish() == 1


def test_a_simulator_that_closes_without_writing_ends_the_loop():
    sim = Simulator(plan)
    sim.send(b'')  # what a simulator that died after opening the pipe leaves
    assert sim.finish() == 0


if __name__ == '__main__':
    for name, test in sorted(globals().items()):
        if name.startswith('test_'):
            test()
            print('ok', name)
