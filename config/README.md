# Configuration

This directory contains configuration files for the project.

## Structure

```
config/
├── api/           # API service configuration files
├── triton/        # Triton server configuration files
└── README.md      # This file
```

## Purpose

Centralized configuration management for:
- Service settings
- Model parameters
- Deployment configurations
- Environment-specific settings

## Configuration Types

### API Configuration (`api/`)
- Server settings (host, port, workers)
- LanguageTool integration settings
- Triton client configuration
- Logging configuration
- Authentication/API keys

### Triton Configuration (`triton/`)
- Model configurations (config.pbtxt files)
- Server startup parameters
- Resource allocation (CPU, GPU, memory)
- Batching and optimization settings

## Configuration Format

Recommended formats:
- **YAML**: Human-readable, good for hierarchical configs
- **JSON**: Standard format, easy to parse
- **TOML**: Python-friendly, good for application configs
- **ENV**: Environment variables for deployment

## Best Practices

1. **Separate environments**: dev, staging, production configs
2. **Secrets management**: Use environment variables or secret managers
3. **Version control**: Track configuration changes
4. **Documentation**: Comment complex settings

## Future Development

- Configuration templates
- Environment-specific config files
- Configuration validation schemas
- Default configuration files
