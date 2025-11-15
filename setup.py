from setuptools import find_packages, setup

setup(
    name='flaschendepot',
    version='0.1.0',
    description='Data Science MLOps Project für Flaschenpfand-Analyse',
    author='Franz',
    author_email='your.email@example.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.9',
    install_requires=[
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
        'mlflow>=2.9.0',
        'pyyaml>=6.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.7.0',
        ],
    },
)
