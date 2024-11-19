from setuptools import setup, find_packages

setup(
    name='photo_g',
    version='0.0.beta',
    description='A Python package for photometry astrometry and image analysis',
    long_description='A Python package for photometry astrometry and image analysis',
    long_description_content_type='text/plain',
    author='Giuliano Margoti',
    author_email='giulianomargoti@on.br',
    url='https://github.com/seu_usuario/photo_g',  
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21',
        'astropy>=4.3.1',
    ],
    python_requires='>=3.7',
    project_urls={
        'Bug Reports': 'https://github.com/seu_usuario/photo_g/issues',  
    },
)
