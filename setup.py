from setuptools import setup, find_packages

setup(
    name="kunal-ai-portfolio",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
    ],
    author="Kunal",
    description="AI Development Portfolio with Learning Resources",
    python_requires=">=3.8",
)
