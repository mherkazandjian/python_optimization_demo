from distutils.core import setup, Extension

module_info = Extension("my_module",
                        include_dirs=[],
                        libraries=[],
                        sources=["mymodule.c"])

setup(name="my_module",
      version="1.0",
      description='This is an example package',
      author='bla bla',
      url='www.blabla.foo',
      ext_modules=[module_info])
