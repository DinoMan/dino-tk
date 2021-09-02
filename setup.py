from setuptools import setup

setup(name='dtk',
      version='0.1',
      description='Dino\'s library for machine learning and other things',
      packages=['dtk', 'dtk.filesystem', 'dtk.metrics', 'dtk.nn','dtk.nn.temporal', 'dtk.transforms', 'dtk.speech', 'dtk.media'],
      package_dir={'dtk': 'dtk'},
      install_requires=[
          'torch',
          'numpy',
          'scikit-image',
          'arrow',
      ],
      zip_safe=False)
