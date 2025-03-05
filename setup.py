from setuptools import setup, find_packages

setup(
    name='DexRep',
    version='1.0',
    description='',
    packages=find_packages(),
    install_requires=[
        'einops==0.8.0',
        'gym==0.26.2',
        'h5py==3.11.0',
        'hydra-core==0.11.3',
        'matplotlib==3.5.1',
        'numpy==1.21.0',
        'point-cloud-utils==0.30.4',
        'tensorboard==2.14.0',
        'tqdm==4.62.3',
        'transforms3d==0.4.2',
        'open3d==0.18.0',
        'trimesh'
    ],
    python_requires='>=3.8',
)