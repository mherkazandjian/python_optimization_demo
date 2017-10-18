from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include


setup_requires = ['setuptools',
                  'Cython',
                  'numpy>=1.7.1']

install_requires = ['numpy>=1.7.1']

# external modules
ext_modules = [
    Extension(
        name="cython_binding",
        sources=["laplacian_filter_wrap.pyx", "../laplacian_filter.c"],
        include_dirs=[get_include()],
        language="c",
        extra_compile_args=['-Ofast', '-fpic'],
    )
]

# installation
setup(
    name='cython_binding',
    packages=find_packages(),
    setup_requires=setup_requires,
    install_requires=install_requires,
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
