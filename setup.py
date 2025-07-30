from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="interpretsr",
    version="1.0.1",
    author="InterpretSR Team",
    description="Neural Networks with Symbolic Regression for Interpretable Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch",
        "pysr",
        "numpy",
        "pandas", 
        "scikit-learn",
        "matplotlib",
        "sympy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
