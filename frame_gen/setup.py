from setuptools import setup, find_packages

setup(
    name="frame_gen",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "opencv-python",
        "scipy",
        "tqdm",
        "pandas",
        "matplotlib",
        "seaborn",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "flake8",
            "mypy",
        ],
        "rife": [
            "numpy>=1.16, <=1.23.5",
            "tqdm>=4.35.0",
            "sk-video>=1.1.10",
            "torch>=1.6.0",
            "opencv-python>=4.1.2",
            "moviepy>=1.0.3",
            "torchvision>=0.7.0",
        ],
        "metrics": [
            "torch>=1.7.0",
            "torchvision>=0.8.0",
            "lpips",
        ],
    },
    python_requires=">=3.8.20",  # Minimum Python version for RIFE
    author="arielgoes",
    author_email="arielgoesdecastro@gmail.com",
    description="Frame generation and degradation simulation toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
) 