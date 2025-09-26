from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="x-ray-ai",
    version="1.0.0",
    author="Dara Mustafa",
    author_email="your.email@example.com",  # Replace with your email
    description="Multi-label chest X-ray disease classification using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dara01x/x-ray-ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.0.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "ipywidgets>=7.0.0",
        ],
    },
    # entry_points={
    #     "console_scripts": [
    #         "x-ray-train=training.cli:train_main",
    #         "x-ray-evaluate=evaluation.cli:evaluate_main", 
    #         "x-ray-inference=utils.cli:inference_main",
    #     ],
    # },
    include_package_data=True,
    zip_safe=False,
)
