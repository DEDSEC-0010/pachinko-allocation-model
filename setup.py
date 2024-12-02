from setuptools import setup, find_packages

setup(
    name="pachinko-allocation-model",
    version="0.1.3",
    author="Josten S Cheeran",
    author_email="jostencheeran@gmail.com",
    description="A probabilistic topic modeling approach inspired by Pachinko Allocation Model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dedsec-0010/pachinko-allocation-model",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy>=1.20.0", "scipy>=1.6.0"],
    extras_require={"dev": ["pytest", "black", "mypy"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
