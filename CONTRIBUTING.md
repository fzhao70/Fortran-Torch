# Contributing to Fortran-Torch

Thank you for your interest in contributing to Fortran-Torch! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the problem
- **Expected behavior** vs actual behavior
- **Environment details** (OS, compiler versions, PyTorch version)
- **Code samples** if applicable
- **Error messages** or logs

Example bug report template:

```markdown
**Environment:**
- OS: Ubuntu 22.04
- Compiler: GCC 11.3.0
- PyTorch: 2.0.1
- LibTorch: 2.0.1 (CPU)

**Description:**
Brief description of the issue

**Steps to Reproduce:**
1. Step one
2. Step two
3. ...

**Expected Behavior:**
What you expected to happen

**Actual Behavior:**
What actually happened

**Error Message:**
```
paste error message here
```
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use case**: Why is this enhancement needed?
- **Proposed solution**: How should it work?
- **Alternatives**: What alternatives have you considered?
- **Examples**: Code examples if applicable

### Contributing Code

We welcome code contributions! Areas where contributions are especially valuable:

- **Bug fixes**
- **New features** (discuss first via issue)
- **Performance improvements**
- **Documentation improvements**
- **Examples** for specific use cases
- **Tests** to improve coverage

## Development Setup

### Prerequisites

1. Install development tools:
   ```bash
   sudo apt install cmake g++ gfortran git
   ```

2. Install Python dependencies:
   ```bash
   pip install torch numpy pytest
   ```

3. Download LibTorch (see [INSTALL.md](INSTALL.md))

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Fortran-Torch.git
   cd Fortran-Torch
   ```

3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/Fortran-Torch.git
   ```

### Build for Development

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DCMAKE_BUILD_TYPE=Debug \
      -DBUILD_EXAMPLES=ON \
      ..
make -j$(nproc)
```

### Keeping Your Fork Updated

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

## Coding Standards

### C++ Code

- **Style**: Follow Google C++ Style Guide
- **Formatting**: Use clang-format (config provided)
- **Comments**: Use Doxygen-style comments for public APIs
- **Error Handling**: Always check return values and handle errors gracefully

Example:

```cpp
/**
 * @brief Load a TorchScript model from file
 * @param model_path Path to the .pt file
 * @param device Device to load model on
 * @param model Output model handle
 * @return Status code
 */
FTorchStatus ftorch_load_model(const char* model_path,
                                FTorchDevice device,
                                FTorchModel* model) {
    if (!model_path || !model) {
        set_error("Null pointer provided");
        return FTORCH_ERROR_NULL_POINTER;
    }
    // ... implementation ...
}
```

### Fortran Code

- **Style**: Follow modern Fortran best practices
- **Indentation**: 4 spaces
- **Line length**: Maximum 100 characters
- **Comments**: Use `!>` for documentation comments
- **Naming**: Use descriptive names, snake_case for procedures

Example:

```fortran
!> Load a TorchScript model
!>
!> @param model_path Path to the model file
!> @param device Device to use (CPU or CUDA)
!> @return Model handle
function torch_load_model(model_path, device) result(model)
    character(len=*), intent(in) :: model_path
    integer(torch_device), intent(in), optional :: device
    type(torch_model) :: model

    ! Implementation
end function torch_load_model
```

### Python Code

- **Style**: Follow PEP 8
- **Type hints**: Use type hints for function signatures
- **Docstrings**: Use Google-style docstrings
- **Testing**: Include tests for new features

Example:

```python
def train_model(input_size: int, output_size: int) -> nn.Module:
    """Train a simple neural network.

    Args:
        input_size: Number of input features
        output_size: Number of output features

    Returns:
        Trained PyTorch model
    """
    # Implementation
    pass
```

### Documentation

- **Markdown**: Use GitHub-flavored markdown
- **Code examples**: Include working code examples
- **API docs**: Document all public APIs
- **Comments**: Explain *why*, not *what*

## Testing

### Running Tests

```bash
cd build
ctest --output-on-failure
```

### Adding Tests

Tests should be added for:
- All new features
- Bug fixes (regression tests)
- Edge cases

Test structure:
```
tests/
├── cpp/           # C++ unit tests
├── fortran/       # Fortran tests
└── integration/   # End-to-end tests
```

Example Fortran test:

```fortran
program test_tensor_creation
    use ftorch
    implicit none

    type(torch_tensor) :: tensor
    real(4) :: data(10)
    integer :: i

    ! Initialize data
    do i = 1, 10
        data(i) = real(i)
    end do

    ! Create tensor
    tensor = torch_tensor_from_array(data)

    ! Test
    if (.not. c_associated(tensor%ptr)) then
        print *, "FAILED: Tensor creation"
        stop 1
    end if

    ! Cleanup
    call torch_free_tensor(tensor)

    print *, "PASSED: Tensor creation"
end program test_tensor_creation
```

## Pull Request Process

### Before Submitting

1. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following coding standards

3. **Test thoroughly**:
   ```bash
   cd build
   make -j$(nproc)
   ctest
   ```

4. **Update documentation** if needed

5. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: brief description

   Longer description of what this commit does and why.

   Fixes #123"
   ```

### Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line should be 50 characters or less
- Reference issues and pull requests after the first line

Examples:
```
Add support for multiple model instances

This commit adds the ability to load and use multiple PyTorch models
simultaneously, which is useful for ensemble methods.

Closes #42
```

### Submitting the Pull Request

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create pull request** on GitHub

3. **Fill out the PR template** with:
   - Description of changes
   - Related issue numbers
   - Testing performed
   - Breaking changes (if any)

4. **Wait for review** and address feedback

### Pull Request Template

```markdown
## Description
Brief description of changes

## Motivation and Context
Why is this change needed? What problem does it solve?

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## How Has This Been Tested?
Describe the tests you ran and their results

## Checklist
- [ ] My code follows the code style of this project
- [ ] I have updated the documentation accordingly
- [ ] I have added tests to cover my changes
- [ ] All new and existing tests passed
- [ ] My changes generate no new warnings
```

### Review Process

- At least one maintainer review required
- CI must pass
- Address all review comments
- Keep PR focused (one feature/fix per PR)
- Be patient and respectful

### After Merge

- Delete your feature branch
- Update your local main branch
- Close related issues if not automatically closed

## Code Review Guidelines

### For Reviewers

- **Be constructive**: Suggest improvements, don't just criticize
- **Be specific**: Point to exact lines and suggest alternatives
- **Be timely**: Review within a few days if possible
- **Be thorough**: Check code, tests, and documentation

### For Authors

- **Be responsive**: Address feedback promptly
- **Be open**: Consider reviewer suggestions
- **Be clear**: Explain your reasoning if you disagree
- **Be patient**: Reviews take time

## Recognition

Contributors will be:
- Listed in the CONTRIBUTORS file
- Acknowledged in release notes
- Thanked in the README

## Questions?

- Open an issue for questions
- Join discussions in GitHub Discussions
- Email: your.email@example.com

Thank you for contributing to Fortran-Torch! 🙏
