from dataclasses import asdict, dataclass

from jinja2 import Environment, FileSystemLoader


@dataclass
class Job:
    dataset: str
    method: str
    duration: str = "7:30:00"
    memory: str = "16G"


def main():
    jobs = [
        Job("cifar100", "01_linear"),
        Job("cifar100", "02_lora"),
        # Job("cifar100", "03_ball"),
        Job("cifar100", "04_replay"),
        # Job("cifar100", "05_gdumb"),
        Job("cifar100", "06_der"),
        # Job("cifar100", "07_joint"),
        Job("cifar100", "08_rwalk"),
    ]
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
