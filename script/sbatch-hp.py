import jinja2
import os
import click

env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(os.path.dirname(__file__)),
    autoescape=jinja2.select_autoescape(),
)

@click.command()
@click.argument("dataset", required=True)
@click.argument("method", required=True)
def main(dataset, method):
    template = env.get_template("hp.sbatch.jinja2")
    rendered = template.render(dataset=dataset, method=method)
    print(rendered)

if __name__ == "__main__":
    main()
