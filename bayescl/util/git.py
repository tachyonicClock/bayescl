from subprocess import check_output


def is_git_status_clean() -> bool:
    output = check_output(["git", "status", "--porcelain"]).decode()
    return output.strip() == ""


def commit_short_hash() -> str:
    output = check_output(["git", "rev-parse", "--short", "HEAD"]).decode()
    return output.strip()
