import os
import signal
import subprocess
import sys
import time


def launch(shell_path, cuda_id, output):
    os.makedirs(output, exist_ok=True)
    print(os.path.join(output, 'output.txt'))
    print(shell_path, cuda_id, output)
    # A launcher that runs `python` gets this interpreter's unless told otherwise: the agents under
    # agents/ that run in the simulator's environment read it, and PATH's `python` is not this one
    # when the simulator was started by its full path without activating the environment.
    env = dict(os.environ)
    env.setdefault('HUGSIM_AGENT_PYTHON', sys.executable)
    with open(os.path.join(output, 'output.txt'), 'w') as f:
        # A session of its own, so kill() reaches the whole launcher: `pixi run` and `conda run`
        # fork the agent's python instead of exec'ing it, and killing the launcher alone left
        # that python blocked on a pipe, holding its GPU memory.
        process = subprocess.Popen(
            ["bash", shell_path, cuda_id, output], stdout=f, stderr=f, env=env, start_new_session=True
        )
    return process


def kill(process):
    """SIGKILL the launcher's process group: the launcher and everything it started."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    process.poll()


def check_alive(process, tolerant=100):
    i = 0
    while i < tolerant:
        return_code = process.poll()
        if return_code is not None:
            print(f"The AD algorithm completed with return code {return_code}.")
            kill(process)
            return
        elif i % 5 == 0:
            print(f"The AD algorithm is still running, remaining tolerant {tolerant - i}.")
        time.sleep(1)
        i += 1
    kill(process)
    print("The AD algorithm process is killed.")
