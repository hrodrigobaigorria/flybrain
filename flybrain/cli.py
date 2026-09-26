"""flybrain command line: verify the dataset, rebuild it, or run the stimulus demo."""

import argparse
import json


def main():
    p = argparse.ArgumentParser(prog="flybrain")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="Check the local connectome dataset against locks")
    sub.add_parser("prepare", help="Download + compile the full dataset (needs source files)")
    d = sub.add_parser("demo", help="Run a non-trading visual stimulus demo")
    d.add_argument("--observations", type=int, default=4)
    a = p.parse_args()

    if a.command == "verify":
        from .data import verify

        print(json.dumps(verify(), indent=2))
    elif a.command == "prepare":
        from .data import prepare

        prepare()
    elif a.command == "demo":
        from .demo import run

        run(a.observations)


if __name__ == "__main__":
    main()
