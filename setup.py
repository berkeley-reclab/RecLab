from setuptools import find_packages, setup


setup(
    name='RecLab',
    version='0.1.1',
    author='Karl Krauth',
    author_email='karl.krauth@gmail.com',
    description='A simulation framework for recommender systems.',
    license='MIT',
    download_url= 'https://github.com/berkeley-reclab/RecLab/archive/v0.1.1.tar.gz',
    packages=find_packages(),
    include_package_data=True,
    url='https://berkeley-reclab.github.io/',
    keywords=[
        'recommender',
        'recommendation',
        'simulation',
        'evaluation'
    ],
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.0.5',
        'scipy>=1.4.1',
    ],
    extras_require={
        'recommenders': [
            'keras>=2.4.3',
            'scikit-learn>=0.23.1',
            'tensorflow>=2.2.0',
            'torch>=1.5.1',
            'wpyfm>=0.1.9',
        ]
    },
    tests_require=[
        'pytest>=5.4.3',
        'pytest-mock>=3.3.0',
    ],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
