# Contributing to TAFC

We welcome contributions to TAFC! This document provides guidelines for contributing.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists in the [GitHub Issues](https://github.com/bijiw515/TAFC/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details (GPU, CUDA version, PyTorch version)
   - Relevant logs or error messages

### Submitting Pull Requests

1. **Fork the repository** and create a new branch from `main`
2. **Make your changes** following our coding standards
3. **Test your changes** thoroughly
4. **Document your changes** in code comments and README if needed
5. **Submit a pull request** with:
   - Clear description of the changes
   - Reference to related issues (if any)
   - Test results showing the changes work

### Coding Standards

- Follow PEP 8 style guidelines for Python code
- Add docstrings to functions and classes
- Keep functions focused and modular
- Add comments for complex logic
- Ensure backward compatibility when possible

### Testing

Before submitting a PR, please:

1. Test on at least one model (FLUX, Wan2.1, or HunyuanVideo)
2. Verify that the speedup and quality metrics are reasonable
3. Check that no existing functionality is broken

### Documentation

- Update README.md if you add new features
- Add usage examples for new parameters
- Update INSTALL.md if dependencies change

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/TAFC.git
cd TAFC

# Create a development branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements-dev.txt  # if available

# Make your changes
# ...

# Test your changes
python test_your_changes.py

# Commit and push
git add .
git commit -m "Description of your changes"
git push origin feature/your-feature-name
```

## Areas for Contribution

We especially welcome contributions in these areas:

1. **New Model Support**: Add TAFC support for other diffusion models
2. **Performance Optimization**: Improve speed or memory efficiency
3. **Evaluation Metrics**: Add new quality metrics or benchmarks
4. **Documentation**: Improve README, add tutorials, or create examples
5. **Bug Fixes**: Fix reported issues
6. **Parameter Tuning**: Share optimal parameter presets for different use cases

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on what is best for the project and community

## Questions?

If you have questions about contributing, feel free to:
- Open a discussion in GitHub Discussions
- Ask in the issue tracker
- Contact the maintainers

Thank you for contributing to TAFC!
