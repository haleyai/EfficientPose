#!/usr/bin/env python3
from setuptools import setup


setup(
    name="efficientpose",
    description="Human pose estimation",
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "tensorflow",  # ==2.5.1
        "pymediainfo==5.0.3",
        "scikit-image==0.17.2",
        "sk-video==1.1.10",
        "opencv-python",
    ],
    extras_require = {
        'torch_models':  ["torch"],
    }    
    packages=['efficientpose', 'efficientpose.utils'],
    package_dir={'efficientpose': '.', 'efficientpose.utils': 'utils'},
    package_data={'efficientpose': ['models/keras/*', 'models/pytorch/*', 'models/tensorflow/*', 'models/tflite/*']}
)
