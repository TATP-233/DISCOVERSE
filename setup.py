from setuptools import find_packages
from distutils.core import setup

setup(
    name="discoverse",
    version="1.6.2",
    author="Yufei Jia",
    license="MIT",
    packages=find_packages(),
    author_email="jyf23@mails.tsinghua.edu.cn",
    description="DISCOVERSE: A photorealistic simulator for Discover robots",
    install_requires=["mujoco", "opencv-python"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
