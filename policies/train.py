if __name__ == "__main__":

    import argparse
    import sys
    import subprocess

    parser = argparse.ArgumentParser(
        description="Run inference on a trained model.", add_help=False
    )
    parser.add_argument(
        "policy",
        type=str,
        default="act",
        help="The type of policy to use.",
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        default=None,
        help="Show this help message and exit.",
    )

    args, unknown = parser.parse_known_args()
    policy = args.policy

    print(f"Running inference with policy: {policy}")
    if args.help:
        parser.print_help()

    args = sys.argv[2:]
    if policy == "act":
        command = f"cd policies/act && python3 policy_train.py "
    elif policy == "dp":
        command = f"cd policies/dp && python3 train_eval.py "
    else:
        raise NotImplementedError(f"Policy {policy} is not implemented.")

    # add args to command
    command += " ".join(args)
    print(f"Running command: {command}")

    try:
        subprocess.run(command, shell=True, check=True)
    except Exception as e:
        pass
