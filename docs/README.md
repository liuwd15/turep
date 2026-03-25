# Documentation

## Getting Started

For practical usage examples, see the [examples](../examples) directory.

## API Reference

### Models

- **FANVI**: scANVI-style model with focal-loss classification.
- **FFADVI**: Factor-disentangled VAE with focal-loss classification.

### Loss Functions

- **FocalLoss**: Multiclass focal-loss module.
- **focal_loss**: Functional interface to compute focal loss.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

This project uses `black` for code formatting:

```bash
black src/ tests/
```

## Contributing

Contributions are welcome. Please ensure:

1. All tests pass
2. Code is formatted with `black`
3. Type hints and docstrings are added where appropriate
