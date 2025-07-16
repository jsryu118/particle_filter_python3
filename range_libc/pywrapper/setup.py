#!/usr/bin/env python3
import os
import platform
import sys
import copy
from os.path import join as pjoin

import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize


def check_for_flag(env_var, truemsg=None, falsemsg=None):
    enabled = os.environ.get(env_var, "").lower() == "on"
    if enabled and truemsg:
        print(truemsg)
    elif not enabled and falsemsg:
        print(falsemsg)
        print(f"   $ export {env_var}=ON && python setup.py install")
    return enabled


use_cuda = check_for_flag(
    "WITH_CUDA",
    truemsg="Compiling with CUDA support",
    falsemsg="Compiling without CUDA support. To enable CUDA use:"
)
trace = check_for_flag(
    "TRACE",
    truemsg="Compiling with trace enabled for Bresenham's Line",
    falsemsg="Compiling without trace enabled for Bresenham's Line"
)

# Mac-specific compile flags
if platform.system().lower() == "darwin":
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = platform.mac_ver()[0]
    os.environ["CC"] = "c++"

def find_in_path(name, path):
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    # Check common install locations or CUDAHOME
    if os.path.isdir("/usr/local/cuda-7.5"):
        home = "/usr/local/cuda-7.5"
    elif os.path.isdir("/usr/local/cuda"):
        home = "/usr/local/cuda"
    elif "CUDAHOME" in os.environ:
        home = os.environ["CUDAHOME"]
    else:
        nvcc = find_in_path("nvcc", os.environ.get("PATH", ""))
        if nvcc is None:
            raise EnvironmentError(
                "The nvcc binary could not be located in your $PATH. "
                "Either add it to your path, or set $CUDAHOME"
            )
        home = os.path.dirname(os.path.dirname(nvcc))

    nvcc = pjoin(home, "bin", "nvcc")
    cudaconfig = {
        "home": home,
        "nvcc": nvcc,
        "include": pjoin(home, "include"),
        "lib64": pjoin(home, "lib64"),
    }
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError(f"The CUDA {k} path could not be located in {v}")
    return cudaconfig

# Compile flags
compiler_flags = ["-w", "-std=c++11", "-march=native", "-ffast-math", "-fno-math-errno", "-O3"]
nvcc_flags     = ["-arch=sm_20", "--ptxas-options=-v", "-c", "--compiler-options", "'-fPIC'", "-w", "-std=c++11"]
include_dirs   = ["../", numpy.get_include()]
depends        = ["../includes/*.h"]
sources        = ["RangeLibc.pyx", "../vendor/lodepng/lodepng.cpp"]

CHUNK_SIZE  = "262144"
NUM_THREADS = "256"

if use_cuda:
    compiler_flags += [f"-DUSE_CUDA=1", f"-DCHUNK_SIZE={CHUNK_SIZE}", f"-DNUM_THREADS={NUM_THREADS}"]
    nvcc_flags     += [f"-DUSE_CUDA=1", f"-DCHUNK_SIZE={CHUNK_SIZE}", f"-DNUM_THREADS={NUM_THREADS}"]

    CUDA = locate_cuda()
    include_dirs.append(CUDA["include"])
    sources.append("../includes/kernels.cu")

if trace:
    compiler_flags.append("-D_MAKE_TRACE_MAP=1")


# Customize the build_ext to handle .cu with nvcc
from setuptools.command.build_ext import build_ext as _build_ext

class custom_build_ext(_build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append(".cu")
        default_compiler_so = self.compiler.compiler_so
        super_compile = self.compiler._compile

        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if os.path.splitext(src)[1] == ".cu":
                self.compiler.set_executable("compiler_so", CUDA["nvcc"])
                postargs = extra_postargs.get("nvcc", [])
            else:
                postargs = extra_postargs.get("gcc", [])
            super_compile(obj, src, ext, cc_args, postargs, pp_opts)
            self.compiler.compiler_so = default_compiler_so

        self.compiler._compile = _compile
        super().build_extensions()

# Extension definition
if use_cuda:
    ext_modules = [
        Extension(
            "range_libc",
            sources,
            include_dirs=include_dirs,
            library_dirs=[CUDA["lib64"]],
            libraries=["cudart"],
            runtime_library_dirs=[CUDA["lib64"]],
            extra_compile_args={"gcc": compiler_flags, "nvcc": nvcc_flags},
            extra_link_args=["-std=c++11"],
            depends=depends,
            language="c++",
        )
    ]
else:
    ext_modules = [
        Extension(
            "range_libc",
            sources,
            include_dirs=include_dirs,
            # non‐CUDA 환경에서도 dict 형태로 전달
            extra_compile_args={"gcc": compiler_flags, "nvcc": []},
            extra_link_args=["-std=c++11"],
            depends=depends,
            language="c++",
        )
    ]

setup(
    name="range_libc",
    version="0.1",
    author="Corey Walsh",
    ext_modules=cythonize(ext_modules),
    cmdclass={"build_ext": custom_build_ext},
)
