# from .triton.triton_templates import
import json
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import yaml
from grammared_language.utils .config_parser import BaseModelConfig, ModelsConfig
from grammared_language.utils import config_parser

DEFAULT_TEMPLATE_FOLDER = str(Path(__file__).parent / "triton_templates")

SUPPORTED_MODEL_TYPES = ("gector", "grammared_classifier", "coedit")

TEMPLATE_FILES_BY_MODEL_TYPE = {
    "gector": {
        "config": "gector.config.pbtxt.jinja",
        "model": "gector.model.py.jinja"
    },
    "grammared_classifier": {
        "config": "grammared_classifier.config.pbtxt.jinja",
        "model": "grammared_classifier.model.py.jinja"
    },
    "coedit": {
        "config": "text2text.config.pbtxt.jinja",
        "model": "text2text.model.py.jinja"
    }
}

class TritonRepoBuilder:
    def __init__(self, template_folder: str|None=None):
        if template_folder is None:
            template_folder = DEFAULT_TEMPLATE_FOLDER
        self.template_folder = template_folder
        self.jina_loader = FileSystemLoader(searchpath=self.template_folder)
        self.jinja_env = Environment(loader=self.jina_loader)

    def build_model_repo(self, repo_folder: str, config: ModelsConfig|None=None, config_path:str|None=None):
        
        if not (config or config_path):
            raise ValueError("Either config or config_path must be provided.")
        
        if config is None:
            config = config_parser.load_config_from_file(config_path)

        for m in config.models:
            model_config = config.models[m]
            if model_config.type not in SUPPORTED_MODEL_TYPES:
                raise ValueError(f"Unsupported model type: {model_config.type}")

            name = m
            if model_config.serving_config.triton_model_name is not None:
                name = model_config.serving_config.triton_model_name

            self._build_model_repo(
                    name=name,
                    model_config=model_config,
                    repo_folder=repo_folder
            )
            
    def _build_model_repo(self, name:str, model_config:BaseModelConfig, repo_folder:str, model_version:int=1):
        model_type = model_config.type
        pretrained_model_name_or_path = model_config.serving_config.pretrained_model_name_or_path
        template_files = TEMPLATE_FILES_BY_MODEL_TYPE[model_type]
        output_folder = Path(repo_folder) / name
        model_folder = output_folder / str(model_version)
        config_file = self.jinja_env.get_template(template_files["config"]).render({
            "model_name": name,
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "json_model_config": json.dumps(model_config.model_dump_json())[1:-1],
        })

        model_file = self.jinja_env.get_template(template_files["model"]).render()
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        Path(model_folder).mkdir(parents=True, exist_ok=True)
        with open(output_folder / "config.pbtxt", "w") as f:
            f.write(config_file)

        with open(model_folder / "model.py", "w") as f:
            f.write(model_file)