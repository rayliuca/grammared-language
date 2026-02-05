# Grammared Language

Adding Grammarly (and other) open source models to LanguageTool

## Demo


![Demo Screenshot](assets/simple-demo.png)


Demo server: [https://grammared-language-demo.rayliu.ca/v2](https://grammared-language-demo.rayliu.ca/v2)


> **Warning:** Demo server is hosted on an Oracle ARM CPU server. It may be slow!


## Overview


## Limitations
- The correction will always show up as grammar corrections
    - LanguageTool does not use the correction categories supplied by the remote servers
- No paraphrasing support
    - LanguageTool clients request to a hard coded rewrite server url


## Supported Models
- `GECToR` models from [gotutiyan/gector](https://github.com/gotutiyan/gector)
- `text2text-generation` models
    - for example, Grammarly's [CoEdIT](https://huggingface.co/collections/grammarly/coedit) models


## Quick Start

### Model Config

#### Config File

See `model_config.yaml` or `docker/default_model_config.yaml` as a template:

```yaml
gector_deberta_large:
    type: gector
    backend: triton
    serving_config:
        triton_host: triton-server
        triton_port: 8001
        pretrained_model_name_or_path: "gotutiyan/gector-deberta-large-5k"
        triton_model_name: gector_deberta_large
        device: cuda # cpu, cuda, or auto
```




#### Environment Variable

Or you can also set via environment variables (see `demo-docker-compose.yml` for real-world examples):


```
# See the 'environment:' section in demo-docker-compose.yml for full model config via env vars
# Example:
GRAMMARED_LANGUAGE__MODELS__GECTOR_DEBERTA_LARGE__TYPE=gector
GRAMMARED_LANGUAGE__MODELS__GECTOR_DEBERTA_LARGE__BACKEND=triton
GRAMMARED_LANGUAGE__MODELS__GECTOR_DEBERTA_LARGE__SERVING_CONFIG__TRITON_HOST=triton-server
...
```


For more, see the comments in `grammared_language/utils/config_parser.py`.

### LanguageTool


To enable remote servers with LanguageTool we will need a remote rule config file, which could be enabled via the `remoteRulesFile` option in the `server.properties` file

---

## How config loading works

When the service starts, it loads model configuration in this order:

1. If a config file exists at `/model_config.yaml`, it loads that.
2. If not, and environment variables starting with `GRAMMARED_LANGUAGE__` are set, it loads config from those (see `demo-docker-compose.yml`).
3. If neither is found, it falls back to `/default_model_config.yaml`.

See the `get_config` function in `grammared_language/utils/config_parser.py` for details.

3. **Start everything with Docker Compose:**
    ```bash
    docker-compose up --build
    ```
    This runs Triton, the API, and (optionally) LanguageTool with remote rules.

---

## Requirements

- Python 3.11+
- Docker (for containers)
- [Triton Inference Server](https://github.com/triton-inference-server/server) (for model serving)
- See `pyproject.toml` for details

## Usage


### LanguageTool Integration

To use remote servers with LanguageTool, set up a remote rule config file (see `example_language_tool_configs/remote-rule-config.json`) and add this to your `server.properties`:

```
remoteRulesFile=./remote-rule-config.json
```

Then run:

```
java -cp languagetool-server.jar org.languagetool.server.HTTPServer --config server.properties
```

#### With Dockerized LanguageTool

If you're using the `meyay/languagetool` or `erikvl87/languagetool` Docker images, set:

```
langtool_remoteRulesFile=<remote file config path in docker>
```

See `docker-compose.yml` for a full example.

## Troubleshooting

- For model loading or inference errors, check Triton and API logs
- For LanguageTool integration, make sure your remote rule config is correct and accessible

## License

See [LICENSE.md](LICENSE.md).

---

## Credits & References

- [Ray Liu](https://github.com/rayliuca) (author/maintainer)
- [GECToR: Grammatical Error Correction: Tag, Not Rewrite](https://github.com/gotutiyan/gector)
- [Grammarly CoEdIT models](https://huggingface.co/collections/grammarly/coedit)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [LanguageTool](https://languagetool.org/)
