from setuptools import setup, find_packages

exec(open('flemme/__version__.py').read())
setup(
    name="flemme",
    packages=find_packages(exclude=["tests"]),
    version=__version__,
    author="Guoqing Zhang, Jingyun Yang",
    license="MIT",
    python_requires='>=3.7', 
    entry_points={'console_scripts': [
        # train
        'train_flemme=flemme.train_flemme:main',    
        # test
        'test_flemme=flemme.test_flemme:main',
        # eval
        'eval_flemme=flemme.eval_flemme:main']
        }
)
