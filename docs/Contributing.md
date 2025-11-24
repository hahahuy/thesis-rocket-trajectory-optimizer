# Contributing Guide

Thank you for your interest in contributing to the Rocket Trajectory Optimizer project! This guide will help you get started.

## Code Style Guidelines

### C++ Code Style

- **Standard**: C++17/C++20
- **Formatting**: Use `clang-format` with Google style (see `.clang-format`)
- **Naming**:
  - Classes: `PascalCase` (e.g., `State`, `Dynamics`)
  - Functions: `camelCase` (e.g., `computeDynamics`, `integrateInterval`)
  - Variables: `camelCase` (e.g., `stateVector`, `timeStep`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_ITERATIONS`)
- **Headers**: Include guards or `#pragma once`
- **Comments**: Use `//` for single-line, `/* */` for multi-line

**Format code before committing:**
```bash
clang-format -i src/**/*.cpp src/**/*.hpp
```

### Python Code Style

- **Formatting**: Use `black` (see `pyproject.toml` for configuration)
- **Linting**: Use `flake8` and `isort` for import sorting
- **Naming**:
  - Classes: `PascalCase` (e.g., `PINN`, `RocketDataset`)
  - Functions: `snake_case` (e.g., `compute_dynamics`, `train_epoch`)
  - Variables: `snake_case` (e.g., `state_vector`, `time_step`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_EPOCHS`)

**Format code before committing:**
```bash
black .
isort .
flake8 .
```

## Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Write/update tests** for your changes:
   - C++: Add tests in `tests/` directory
   - Python: Add tests in `tests/` directory with `test_` prefix

4. **Run tests** to ensure everything passes:
   ```bash
   # C++ tests
   cd build && ctest -C Release
   
   # Python tests
   pytest tests/ -v
   ```

5. **Update documentation** if your changes affect:
   - Architecture: Update `ARCHITECTURE_CHANGELOG.md`
   - API: Update relevant WP comprehensive description
   - Setup: Update `SETUP.md` or `SETUP_ENVIRONMENT.md`

6. **Commit your changes** with clear commit messages:
   ```bash
   git commit -m "Add feature: brief description"
   ```

7. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a pull request on GitHub with:
   - Clear description of changes
   - Reference to related issues (if any)
   - Test results

## Testing Requirements

### C++ Tests

- Use GoogleTest framework
- Place tests in `tests/` directory
- Test file naming: `test_*.cpp`
- Run tests: `ctest -C Release` from build directory

### Python Tests

- Use pytest framework
- Place tests in `tests/` directory
- Test file naming: `test_*.py`
- Test function naming: `test_*`
- Run tests: `pytest tests/ -v`

### Test Coverage

- Aim for ≥80% code coverage
- Critical paths (dynamics, constraints, OCP solver) should have ≥90% coverage
- Coverage reports are generated automatically in CI

## Documentation Standards

### Code Documentation

- **C++**: Use Doxygen-style comments for public APIs
- **Python**: Use docstrings (Google style) for all public functions/classes

**Example (Python):**
```python
def compute_dynamics(state, control, params):
    """
    Compute 6-DOF rocket dynamics.
    
    Args:
        state: State vector [14]
        control: Control vector [5]
        params: Physical parameters
        
    Returns:
        State derivative [14]
    """
    # Implementation
```

### Architecture Changes

When adding new PINN architectures or significant changes:

1. **Update `ARCHITECTURE_CHANGELOG.md`** with:
   - Date and architecture name
   - Added/modified files
   - Architecture details
   - Configuration examples
   - Implementation status

2. **Update `architecture_diagram.md`** with Mermaid diagram if it's a new architecture

3. **Update relevant WP comprehensive description** if it affects that work package

### Setup Documentation

When adding new dependencies or changing setup:

1. **Update `SETUP_ENVIRONMENT.md`** with installation instructions
2. **Update `SETUP.md`** if it affects the quick start
3. **Update `requirements.txt`** or `environment.yml` for Python dependencies
4. **Update `CMakeLists.txt`** for C++ dependencies

## Development Workflow

### Setting Up Development Environment

1. Follow [SETUP.md](SETUP.md) for initial setup
2. Install development dependencies:
   ```bash
   pip install black flake8 isort pytest pytest-cov
   ```

### Making Changes

1. **Create feature branch** from `main`
2. **Make incremental commits** with clear messages
3. **Run tests frequently** during development
4. **Format code** before committing
5. **Update documentation** as you go

### Before Submitting PR

- [ ] All tests pass
- [ ] Code is formatted (black, clang-format)
- [ ] No linter errors (flake8, clang-tidy)
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] PR description is complete

## Issue Reporting

When reporting issues, please include:

1. **Description**: Clear description of the problem
2. **Steps to Reproduce**: Minimal steps to reproduce
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: OS, Python version, dependencies versions
6. **Error Messages**: Full error traceback (if any)
7. **Screenshots/Logs**: If applicable

## Questions?

- Check existing documentation in `docs/`
- Review [DESIGN.md](DESIGN.md) for architecture overview
- Check [SETUP.md](SETUP.md) for setup issues
- Open an issue for questions or discussions

Thank you for contributing! 🚀

