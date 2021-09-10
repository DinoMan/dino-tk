from setuptools import setup

setup(name='dtk',
      version='0.2',
      description='Dino\'s library for machine learning and other things',
      author='Dino Vougioukas',
      author_email='dinovgk@gmail.com',
      url='https://github.com/DinoMan/dino-tk.git',
      download_url='https://github.com/DinoMan/dino-tk/archive/refs/tags/v0.2.tar.gz',
      keywords=['machine-learning', 'toolkit', 'file operations'],
      license='MIT',
      packages=['dtk', 'dtk.filesystem', 'dtk.metrics', 'dtk.nn', 'dtk.nn.temporal', 'dtk.transforms', 'dtk.speech', 'dtk.media'],
      package_dir={'dtk': 'dtk'},
      install_requires=[
          'torch',
          'numpy',
          'scikit-image',
          'arrow',
      ],
      zip_safe=False)
