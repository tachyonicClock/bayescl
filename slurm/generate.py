import csv
from dataclasses import asdict, dataclass

from jinja2 import Environment, FileSystemLoader


@dataclass
class Job:
    dataset: str
    method: str
    duration: str
    memory: str


def main():
    # Load slurm/jobs.csv as Job objects
    jobs = []
    with open("slurm/jobs.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            job = Job(**row)
            jobs.append(job)

    env = Environment(loader=FileSystemLoader("slurm"))
    template = env.get_template("template.sl.jinja")
    for job in jobs:
        script_content = template.render(asdict(job))
        script_filename = f"sbatch/{job.dataset}_{job.method}.sl"
        with open(script_filename, "w") as f:
            f.write(script_content)
        print(f"+ {script_filename}")


if __name__ == "__main__":
    main()
