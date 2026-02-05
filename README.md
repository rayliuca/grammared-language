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
<instructions on how to config via config file>


#### Environment Variable

<instructions on how to config via env var>


### LanguageTool


To enable remote servers with LanguageTool we will need a remote rule config file, which could be enabled via the `remoteRulesFile` option in the `server.properties` file

i.e.:
```
remoteRulesFile=./remote-rule-config.json
```


and then:

```
java -cp languagetool-server.jar org.languagetool.server.HTTPServer --config server.properties
```

### Docker


When using the `meyay/languagetool` docker image, add the environment variable:

```
langtool_remoteRulesFile=<remote file config path in docker>
```

See the `docker-compose.yml` file for example



## License

To be determined.
