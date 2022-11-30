from setuptools import setup, find_packages

setup(
    name='arroz',
    packages=find_packages(),
    description='Arroz Con Cosas',
    version='0.0.1',
    url='https://github.com/pabloppp/Arroz-Con-Cosas',
    author='Pablo Pernías',
    author_email='pablo@pernias.com',
    keywords=['pip', 'pytorch', 'generative', 'ldm', 'vqgan'],
    zip_safe=False,
    install_requires=[
        'torch~=1.6',
        'torchvision',
        'numpy==1.*',
        'git+https://github.com/pabloppp/pytorch-tools git+https://github.com/fbcotter/pytorch_wavelets~=1.3',
        'PyWavelets~=1.3',
        'open_clip_torch~=2.7'
    ],
    package_data={},
    include_package_data=True,
)