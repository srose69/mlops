# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Shared by every package's setup.py: target architectures, the dependencies' headers, and a build_ext that
builds the package's static library with its Makefile (parallel, every architecture in one go) before the
PyTorch extension links it.

Variables (all optional):
  MLOPS_ARCHS="61 89"            architectures, or TORCH_CUDA_ARCH_LIST="6.1;8.9"; default: the GPUs visible to
                                 PyTorch, or all of sm_50..sm_90 if none is. sm_89 needs nvcc 12.1+.
  MLOPS_CCBIN, MAX_JOBS          host compiler for nvcc, parallel jobs
  MLOPS_OBJDIR                   object directory that already holds the library for these architectures
                                 (dev/release.sh builds it once for every wheel)
  MLOPS_LOCAL_VERSION            local version label of a release wheel (e.g. cu126torch2.7)
"""
import importlib.util
import os
import re
import shutil
import subprocess
import sys

SUPPORTED = [50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90]  # GPUs without native FP4
RELEASES = {"nvfp4effma": "https://github.com/srose69/nvfp4effma/releases",
            "e5m2effma": "https://github.com/srose69/e5m2effma/releases"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LICENSE_FILES = ["LICENSE", "LICENSE-EXCEPTION", "NOTICE"]


def fail(name, msg):
    sys.exit(f"{name}: {msg}")


def header_version(path, macro):
    m = re.search(rf'#define {macro}_VERSION "([^"]+)"', open(path).read())
    return m.group(1) if m else None


def nvcc_version(cuda_home):
    out = subprocess.check_output([os.path.join(cuda_home, "bin", "nvcc"), "--version"], text=True)
    m = re.search(r"release (\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def target_archs(name, cuda_home):
    import torch
    env = os.environ.get("MLOPS_ARCHS") or os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env:
        archs = set()
        for tok in re.split(r"[;,\s]+", env.strip()):
            m = re.fullmatch(r"(\d+)\.?(\d)?(\+PTX)?", tok)
            if not m:
                fail(name, f"cannot parse architecture '{tok}' (expected e.g. 61 or 6.1)")
            archs.add(int(m.group(1) + (m.group(2) or "")))
        source = "MLOPS_ARCHS / TORCH_CUDA_ARCH_LIST"
    elif torch.cuda.is_available() and torch.cuda.device_count():
        archs = {10 * a + b for a, b in (torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))}
        source = "visible GPUs"
    else:
        archs, source = set(SUPPORTED), "default (no GPU visible)"
    dropped = sorted(a for a in archs if a not in SUPPORTED)
    if dropped:
        print(f"{name}: skipping sm_{', sm_'.join(map(str, dropped))} (native FP4 or unsupported)", file=sys.stderr)
    archs = sorted(a for a in archs if a in SUPPORTED)
    if 89 in archs and nvcc_version(cuda_home) < (12, 1):
        print("%s: skipping sm_89: its FP8 cvt needs nvcc 12.1+ (PTX ISA 8.1), found %d.%d" % ((name,) + nvcc_version(cuda_home)),
              file=sys.stderr)
        archs.remove(89)
    if not archs:
        fail(name, f"no supported architecture to build for ({source}); mlops targets sm_50..sm_90")
    print(f"{name}: building for sm_{', sm_'.join(map(str, archs))} ({source})", file=sys.stderr)
    return archs


def dependency(name, pkg, why):
    """(directory with include/ and src/, version) of an installed EffMA package, found without importing it
    (importing e5m2effma changes torch's _scaled_mm). Nothing is downloaded: a missing package stops the build
    with where to get it."""
    spec = importlib.util.find_spec(pkg)
    where = f"install it first from {RELEASES[pkg]} (pip install <the wheel for your Python / torch / CUDA>)"
    if spec is None or not spec.submodule_search_locations:
        fail(name, f"the {pkg} package is not installed; it is needed {why}: {where}")
    d = spec.submodule_search_locations[0]
    h = os.path.join(d, "include", f"{pkg}.h")
    if not (os.path.isfile(h) and os.path.isfile(os.path.join(d, "src", f"{pkg}.cuh"))):
        fail(name, f"the installed {pkg} has no headers (they ship from {pkg} 0.4 / 0.2 on): {where}")
    return d, header_version(h, pkg.upper())


def local_version(version):
    lv = os.environ.get("MLOPS_LOCAL_VERSION")
    return f"{version}+{lv}" if lv else version


def copy_license_files(dest):
    """The license files live at the repository root; wheels must carry them (setuptools only takes license
    files from the package directory)."""
    for f in LICENSE_FILES:
        if os.path.isfile(os.path.join(ROOT, f)):
            shutil.copyfile(os.path.join(ROOT, f), os.path.join(dest, f))
    return LICENSE_FILES


def make_build(name, pkg_dir, lib, archs, deps):
    """build_ext that first runs `make -C pkg_dir <objdir>/lib<lib>.a` for archs, with the dependencies'
    directories (deps: {MAKE_VARIABLE: dir}), and links the library into every extension."""
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension

    class Build(BuildExtension):
        def run(self):
            objdir = os.path.abspath(os.environ.get("MLOPS_OBJDIR") or os.path.join(self.build_temp, f"lib{lib}"))
            cmd = ["make", "-C", pkg_dir, f"OBJDIR={objdir}", f"ARCHS={' '.join(map(str, archs))}",
                   f"CUDA_HOME={CUDA_HOME}", f"{objdir}/lib{lib}.a"] + [f"{k}={v}" for k, v in deps.items() if v]
            if os.environ.get("MLOPS_CCBIN"):
                cmd.append(f"CCBIN={os.environ['MLOPS_CCBIN']}")
            if os.environ.get("MAX_JOBS"):
                cmd.append(f"JOBS={os.environ['MAX_JOBS']}")
            print(" ".join(cmd), file=sys.stderr)
            subprocess.check_call(cmd)
            for ext in self.extensions:
                ext.extra_objects.append(os.path.join(objdir, f"lib{lib}.a"))
            super().run()

    if CUDA_HOME is None:
        fail(name, "CUDA toolkit not found; set CUDA_HOME")
    return Build
