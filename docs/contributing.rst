Contributing
============

We welcome contributions to InterpretSR! This document provides guidelines for contributing to the project.

Development Setup
----------------

1. Fork the repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

   git clone https://github.com/yourusername/InterpretSR.git
   cd InterpretSR

3. Create a virtual environment and install dependencies:

.. code-block:: bash

   python -m venv interpretsr_venv
   source interpretsr_venv/bin/activate  # On Windows: interpretsr_venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available

4. Install the package in development mode:

.. code-block:: bash

   pip install -e .

How to Contribute
----------------

Types of Contributions
~~~~~~~~~~~~~~~~~~~~~

- **Bug Reports**: Found a bug? Please report it!
- **Feature Requests**: Have an idea for a new feature? Let us know!
- **Code Contributions**: Bug fixes, new features, performance improvements
- **Documentation**: Improvements to docs, examples, or tutorials
- **Tests**: Additional test cases to improve coverage

Reporting Issues
~~~~~~~~~~~~~~~

When reporting issues, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior  
- Your environment details (Python version, OS, etc.)
- Minimal code example if applicable

Pull Request Process
~~~~~~~~~~~~~~~~~~~

1. Create a new branch for your feature/fix:

.. code-block:: bash

   git checkout -b feature/your-feature-name

2. Make your changes and add tests if applicable
3. Ensure all tests pass:

.. code-block:: bash

   python -m pytest

4. Update documentation if needed
5. Commit your changes with a clear message:

.. code-block:: bash

   git commit -m "Add feature: brief description"

6. Push to your fork and create a pull request

Code Style Guidelines
-------------------

- Follow PEP 8 for Python code style
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and concise
- Add type hints where appropriate

Example of good docstring style:

.. code-block:: python

   def example_function(param1: int, param2: str = "default") -> bool:
       """
       Brief description of what the function does.
       
       Args:
           param1 (int): Description of param1
           param2 (str, optional): Description of param2. Defaults to "default".
           
       Returns:
           bool: Description of return value
           
       Raises:
           ValueError: When param1 is negative
           
       Example:
           >>> result = example_function(5, "test")
           >>> print(result)
           True
       """
       if param1 < 0:
           raise ValueError("param1 must be non-negative")
       return param1 > 0 and param2 != ""

Testing Guidelines
-----------------

- Write unit tests for new functionality
- Ensure tests are independent and can run in any order
- Use descriptive test names that explain what is being tested
- Include both positive and negative test cases
- Test edge cases and error conditions

Example test structure:

.. code-block:: python

   import pytest
   import torch
   from InterpretSR.src.mlp_sr import MLP_SR
   
   class TestMLPSR:
       def test_init_with_name(self):
           """Test MLP_SR initialization with custom name."""
           mlp = torch.nn.Linear(5, 1)
           wrapper = MLP_SR(mlp, "test_model")
           assert wrapper.mlp_name == "test_model"
           
       def test_init_without_name(self):
           """Test MLP_SR initialization without custom name."""
           mlp = torch.nn.Linear(5, 1)
           wrapper = MLP_SR(mlp)
           assert wrapper.mlp_name.startswith("mlp_")
           
       def test_forward_pass(self):
           """Test basic forward pass functionality."""
           mlp = torch.nn.Sequential(
               torch.nn.Linear(3, 10),
               torch.nn.ReLU(),
               torch.nn.Linear(10, 1)
           )
           wrapper = MLP_SR(mlp, "test")
           x = torch.randn(5, 3)
           output = wrapper(x)
           assert output.shape == (5, 1)

Documentation Guidelines
-----------------------

- Write clear, concise documentation
- Include practical examples
- Update docstrings when changing function signatures
- Add new modules to the API reference
- Consider adding examples for complex features

Building Documentation Locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build and preview documentation changes:

.. code-block:: bash

   cd docs
   make html
   # Open _build/html/index.html in your browser

Release Process
--------------

(For maintainers)

1. Update version numbers in relevant files
2. Update CHANGELOG.md with new features and fixes  
3. Create a new release on GitHub
4. Build and upload to PyPI:

.. code-block:: bash

   python setup.py sdist bdist_wheel
   twine upload dist/*

Community Guidelines
-------------------

- Be respectful and constructive in discussions
- Help newcomers and answer questions when possible
- Follow the code of conduct
- Give credit where credit is due
- Focus on the technical merits of ideas

Getting Help
-----------

If you need help contributing:

- Check existing issues and discussions
- Read through the codebase and documentation
- Ask questions in issues or discussions
- Reach out to maintainers if needed

Thank you for contributing to InterpretSR!