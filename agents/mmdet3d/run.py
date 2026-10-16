"""Serve UniAD or VAD to closed_loop.py. Normally started through uniad.sh / vad.sh.

    python run.py --agent uniad|vad --ckpt CKPT --output EPISODE_DIR [--device cuda|cpu]
"""

import argparse
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load_client():
    """agents/client.py by path: putting the repo root on sys.path would shadow the vendored trees."""
    spec = importlib.util.spec_from_file_location('hugsim_agent_client', os.path.join(HERE, '..', 'client.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--agent', required=True, choices=['uniad', 'vad'])
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--output', required=True, help='the episode folder closed_loop.py passes')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    client = load_client()
    output = os.path.abspath(args.output)  # the runner changes directory
    import runner

    agent = runner.load_agent(args.agent, args.ckpt, device=args.device, output=output)
    client.serve(output, agent.step, on_done=agent.finish, name=args.agent)


if __name__ == '__main__':
    main()
