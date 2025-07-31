import os
import torch
from setuptools import setup
from torch.utils import cpp_extension

sources = ["src/ProcessGroupULFM.cpp"]
mpi_home = "/home/ziyueliu/openmpi-5.0.8-install"
include_dirs = [f"{os.path.dirname(os.path.abspath(__file__))}/include/", f"{mpi_home}/include"]
library_dirs = [f"{mpi_home}/lib"]

if torch.cuda.is_available():
    module = cpp_extension.CUDAExtension(
        name="ulfm_collectives",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs
    )
else:
    module = cpp_extension.CppExtension(
        name="ulfm_collectives",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs
    )

setup(
    name="ulfm_collectives",
    version="0.0.1",
    ext_modules=[module],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
