"""The AD side of closed_loop.py's FIFO protocol, shared by every agent under agents/.

closed_loop.py launches an agent as ``bash <launcher> <cuda_id> <episode_dir>`` and talks to it
over two FIFOs in the episode folder. Each step it writes ``pickle.dumps((obs, info))`` to
``obs_pipe`` and blocks reading ``plan_pipe`` for the answer; when the episode ends it writes
``pickle.dumps('Done')`` to ``obs_pipe`` instead. Nothing watches the agent's process meanwhile,
so an agent that dies without answering leaves the simulator blocked on ``plan_pipe``, and one
that exits after answering ``None`` leaves it blocked opening ``obs_pipe`` to send ``'Done'``.

:func:`serve` keeps both promises whatever the agent does: every request gets an answer (``None``
when the step raised, which closed_loop.py takes as the AD side crashing and ends the episode),
and the loop always reads ``'Done'`` before returning. The upstream clients (UniAD_SIM, VAD_SIM,
NAVSIM) each carried their own copy of this loop, and on an exception in inference two of them
crashed before answering.

Standard library and numpy only: it is imported in every agent environment.
"""

import os
import pickle
import sys
import traceback


def make_pipes(episode_dir):
    """Create ``obs_pipe`` and ``plan_pipe`` in the episode folder if the simulator has not yet."""
    os.makedirs(episode_dir, exist_ok=True)
    pipes = []
    for name in ('obs_pipe', 'plan_pipe'):
        path = os.path.join(episode_dir, name)
        if not os.path.exists(path):
            try:
                os.mkfifo(path)
            except FileExistsError:  # the simulator made it in between
                pass
        pipes.append(path)
    return pipes


def answer(step, obs, info, episode_dir, name='agent', index=0):
    """``step(obs, info)``, or ``None`` when it raises: what an agent answers one request with.

    The traceback is printed and kept in ``<episode_dir>/agent_crash.txt``. Shared by
    :func:`serve` and closed_loop.py's ``--in_process`` mode, so an agent fails the same way
    whichever of the two drives it.
    """
    try:
        return step(obs, info)
    except Exception:
        trace = traceback.format_exc()
        print(f'[{name}] step {index} raised; answering None\n{trace}', file=sys.stderr, flush=True)
        with open(os.path.join(episode_dir, 'agent_crash.txt'), 'w') as f:
            f.write(trace)
        return None


def serve(episode_dir, step, on_done=None, name='agent'):
    """Answer the simulator's requests with ``step(obs, info)`` until it sends ``'Done'``.

    ``step`` returns what closed_loop.py accepts: an ``(N, 2)`` plan in the ego frame,
    ``(right, forward)`` in metres, waypoints 0.5 s apart; a dict carrying such a ``'plan'`` and
    an override (see closed_loop.py); or ``None`` to end the episode. A step that raises is
    answered ``None``, its traceback printed (the agent's stdout and stderr are the episode's
    ``output.txt``) and kept in ``<episode_dir>/agent_crash.txt``. After a ``None`` the loop goes
    on to read the ``'Done'`` the simulator then sends. ``on_done()`` runs once, after ``'Done'``.
    Returns the number of steps answered with a plan.
    """
    obs_pipe, plan_pipe = make_pipes(episode_dir)
    print(f'[{name}] ready on {episode_dir}', flush=True)
    planned = 0
    while True:
        with open(obs_pipe, 'rb') as pipe:
            raw = pipe.read()
        if not raw:
            # The simulator closed the pipe without writing: it is gone, and an answer would
            # wait forever for a reader. There is no 'Done' coming either.
            print(f'[{name}] the simulator closed the pipe without a request; stopping', flush=True)
            return planned
        # A request this environment cannot unpickle (a numpy 2.x simulator talking to a
        # numpy 1.x agent, say) is answered None like a step that raised, not left unanswered.
        request = answer(lambda *_: pickle.loads(raw), None, None, episode_dir, name, planned)
        if isinstance(request, str) and request == 'Done':
            break
        reply = None if request is None else answer(step, *request, episode_dir, name, planned)
        with open(plan_pipe, 'wb') as pipe:
            pipe.write(pickle.dumps(reply))
        if reply is not None:
            planned += 1
    print(f'[{name}] Done after {planned} planned steps', flush=True)
    if on_done is not None:
        on_done()
    return planned
