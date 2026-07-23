from subprocess import check_output


def commit_short_hash() -> str:
    output = check_output(["git", "rev-parse", "--short", "HEAD"]).decode()
    return output.strip()


def commit_message() -> str:
    output = check_output(["git", "log", "-1", "--pretty=%B"]).decode()
    return output.strip()
