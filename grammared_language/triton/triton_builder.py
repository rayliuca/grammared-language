# from .triton.triton_templates import
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

class TritonBuilder:
    def __init__(self, template_folder: str="./triton_templates"):
        self.template_folder = template_folder
        self.jina_loader = FileSystemLoader(searchpath=self.template_folder)
        self.jinja_env = Environment(loader=self.jina_loader)

    def build_gector_repo(self, name:str, pretrained_model_name_or_path:str, repo_folder:str):
        output_folder = Path(repo_folder) / name
        model_folder = output_folder / "1"
        config_file = self.jinja_env.get_template("gector_config.pbtxt.jinja").render({
            "model_name": name,
            "pretrained_model_name_or_path": pretrained_model_name_or_path
        })

        model_file = self.jinja_env.get_template("gector_model.py.jinja").render()
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        Path(model_folder).mkdir(parents=True, exist_ok=True)
        with open(output_folder / "config.pbtxt", "w") as f:
            f.write(config_file)

        with open(model_folder / "model.py", "w") as f:
            f.write(model_file)


    def build_grammared_classifier_repo(self, name:str, pretrained_model_name_or_path:str, repo_folder:str):
        output_folder = Path(repo_folder) / name
        model_folder = output_folder / "1"
        config_file = self.jinja_env.get_template("grammared_classifier_config.pbtxt.jinja").render({
            "model_name": name,
            "pretrained_model_name_or_path": pretrained_model_name_or_path
        })

        model_file = self.jinja_env.get_template("grammared_classifier_model.py.jinja").render()
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        Path(model_folder).mkdir(parents=True, exist_ok=True)
        with open(output_folder / "config.pbtxt", "w") as f:
            f.write(config_file)

        with open(model_folder / "model.py", "w") as f:
            f.write(model_file)