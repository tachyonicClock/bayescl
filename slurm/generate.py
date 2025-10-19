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
        # CIFAR100
        Job("cifar100", "01_linear"),
        Job("cifar100", "02_lora"),
        Job("cifar100", "03_ball"),
        Job("cifar100", "04_replay"),
        # Job("cifar100", "05_gdumb"),
        Job("cifar100", "06_der"),
        # Job("cifar100", "07_joint"),
        Job("cifar100", "08_rwalk"),
        # DOMAINNET
        Job("domainnet", "01_linear", duration="14:00:00"),
        Job("domainnet", "02_lora", duration="14:00:00"),
        Job("domainnet", "03_ball", duration="14:00:00"),
        Job("domainnet", "04_replay", duration="14:00:00"),
        # Job("domainnet", "05_gdumb", duration="14:00:00"),
        Job("domainnet", "06_der", duration="14:00:00"),
        # Job("domainnet", "07_joint", duration="14:00:00"),
        Job("domainnet", "08_rwalk", duration="14:00:00"),
        # IMAGENET-R
        Job("imagenetr", "01_linear", duration="08:00:00"),
        Job("imagenetr", "02_lora", duration="08:00:00"),
        Job("imagenetr", "03_ball", duration="08:00:00"),
        Job("imagenetr", "04_replay", duration="08:00:00"),
        # Job("imagenetr", "05_gdumb", duration="08:00:00"),
        Job("imagenetr", "06_der", duration="08:00:00"),
        # Job("imagenetr", "07_joint", duration="08:00:00"),
        Job("imagenetr", "08_rwalk", duration="08:00:00"),
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
