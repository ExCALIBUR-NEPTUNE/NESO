import subprocess
import sys
import os


def get_git_hash():
    """
    :returns: Current repository git hash.
    """
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"])
        .decode(sys.stdout.encoding)
        .strip()
    )


def get_run_command():
    """
    :returns: The command issued to ultimately invoke this function.
    """

    return os.path.basename(sys.executable) + " " + " ".join(sys.argv[:])


def get_generation_header():
    """
    :returns: A generic header to describe the git hash and command used to
    generate the output.
    """
    h = get_git_hash()
    cmd = get_run_command()

    m = f"""This is a generated file. Please make non-ephemeral changes by
modifying the script which generates this file. This file was generated on git
hash

{h}

by running the command

{cmd}
"""
    return m


if __name__ == "__main__":
    print(get_generation_header())
