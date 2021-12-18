from setuptools import setup, find_packages

with open("requirements.txt", "r") as requirements_file:
    requirements = list(map(lambda line: line.strip(), requirements_file.readlines()))
    requirements = list(filter(lambda line: (not line.startswith("#")) and len(line) > 0, requirements))

setup(name="redvox-pandas",
      version="1.3.3",
      url='https://github.com/RedVoxInc/redpandas',
      license='Apache',
      author='RedVox',
      author_email='dev@redvoxsound.com',
      description='Library to streamline preprocessing of RedVox API 900 and API 1000 data',
      packages=find_packages(include=[
          "redpandas",
          "redpandas.redpd_plot"
      ],
          exclude=['tests']),
      long_description_content_type='text/markdown',
      long_description=open('README.md').read(),
      install_requires=requirements,
      python_requires=">=3.7")
