from setuptools import setup

setup(name='pybiv',
      version='0.1',
      description='Working with sums of bivariate functions in Python.',
      url='http://github.com/NiMlr/pybiv',
      author='Nils Müller',
      license='MIT',
      packages=['pybiv', 'pybiv.test', 'pybiv.approximate', 'pybiv.optimize', 'pybiv.tools'],
      install_requires=['numpy', 'scipy'],
      zip_safe=False,
      package_data={'': ['testdata.npy']},
      include_package_data=True)
