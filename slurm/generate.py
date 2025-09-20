from dataclasses import asdict, dataclass

from jinja2 import Environment, FileSystemLoader


@dataclass
class Job:
    dataset: str
    method: str
    duration: str = "8:00:00"
    memory: str = "16G"
    args: str = ""


def main():
    jobs = [
        Job("cifar100", "01_linear"),
        Job("cifar100", "02_lora"),
        Job("cifar100", "03_ball", args="peft.save=True"),
        # Job("cifar100", "04_replay"),
        Job("cifar100", "08_rwalk"),
        Job("cifar100", "07_joint"),
        Job("cifar100", "04_replay"),
        Job("domainnet", "01_linear", duration="12:00:00"),
        Job("domainnet", "02_lora", duration="12:00:00"),
        Job("domainnet", "03_ball", duration="12:00:00", args="peft.save=True"),
        # Job("domainnet", "04_replay", duration="12:00:00"),
        Job("domainnet", "07_joint", duration="12:00:00"),
        Job("domainnet", "08_rwalk", duration="12:00:00"),
        Job("imagenetr", "01_linear"),
        Job("imagenetr", "02_lora"),
        Job("imagenetr", "03_ball", args="peft.save=True"),
        # Job("imagenetr", "04_replay"),
        Job("imagenetr", "07_joint"),
        Job("imagenetr", "08_rwalk"),
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
