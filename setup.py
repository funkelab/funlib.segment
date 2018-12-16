from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension

setup(
        name='segment',
        version='0.1',
        description='Tools to segment graphs and volumes.',
        url='https://github.com/funkelab/segment',
        author='Jan Funke',
        author_email='funkej@janelia.hhmi.org',
        license='MIT',
        packages=[
            'segment',
            'segment.graphs',
            'segment.graphs.impl',
            'segment.arrays',
            'segment.arrays.impl'
        ],
        ext_modules=[
            Extension(
                'segment.arrays.impl.replace_values_inplace',
                sources=[
                    'segment/arrays/impl/replace_values_inplace.pyx'
                ],
                extra_compile_args=['-O3'],
                language='c++'),
            Extension(
                'segment.graphs.impl.connected_components',
                sources=[
                    'segment/graphs/impl/connected_components.pyx',
                    'segment/graphs/impl/connected_components_impl.cpp'
                ],
                extra_compile_args=['-O3', '-std=c++11'],
                language='c++')
        ],
        cmdclass={'build_ext': build_ext}
)
