"""Compiles bitpattern.bdd from src/lib.mojo, for wheels and editable installs."""

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# The Mojo runtime library the module loads, and the libraries it loads.
RUNTIME = [
    "libKGENCompilerRTShared",
    "libAsyncRTRuntimeGlobals",
    "libMSupportGlobals",
]
# The module runs on any CPU of the architecture, not just the build machine's.
TARGET_CPUS = {
    "x86_64": "x86-64-v2",
    "aarch64": "generic",  # any ARMv8-A, eg. Graviton or Raspberry Pi
    "arm64": "apple-m1",  # every Apple Silicon Mac
}
if sys.platform == "darwin":
    # macOS 13, the oldest that mojo-compiler runs on. The triple sets it for the
    # compiler, and MACOSX_DEPLOYMENT_TARGET for the linker.
    PLATFORM_TAG = "macosx_13_0_arm64"
    TARGET_FLAGS = ["--target-triple", "arm64-apple-macosx13.0"]
    BUILD_ENV = {"MACOSX_DEPLOYMENT_TARGET": "13.0"}
    # Python's symbols resolve when Python loads the module, as on Linux.
    LINK_FLAGS = ["-Xlinker", "-undefined", "-Xlinker", "dynamic_lookup"]
    # A fixed install name, so a build doesn't depend on where it ran.
    LINK_FLAGS += ["-Xlinker", "-install_name", "-Xlinker", "@rpath/bdd.abi3.so"]
    STRIP_FLAGS = ["-Xlinker", "-x"]  # drops local symbols
    LIBRARY_SUFFIX = ".dylib"
    ORIGIN = "@loader_path"
else:
    # The module needs glibc 2.34 (for dlsym), as does mojo-compiler.
    PLATFORM_TAG = f"manylinux_2_34_{platform.machine()}"
    TARGET_FLAGS = []
    BUILD_ENV = {}
    LINK_FLAGS = []
    STRIP_FLAGS = ["-Xlinker", "--strip-all"]
    LIBRARY_SUFFIX = ".so"
    ORIGIN = "$ORIGIN"


class MojoBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        root = Path(self.root)
        flags = ["-I", str(root / "src"), "-D", f"VERSION={self.metadata.version}"]
        # Source paths in the module are the project's, so a build doesn't depend on
        # where it ran.
        flags += ["-strip-file-prefix", f"{root}/"]
        flags += TARGET_FLAGS + LINK_FLAGS
        if cpu := TARGET_CPUS.get(platform.machine()):
            flags += ["--target-cpu", cpu]
        if version == "editable":
            # The module stays in the project, away from site-packages, so it gets
            # copies of the runtime libraries.
            module = root / "bitpattern" / "bdd.abi3.so"
            # The build environment's mojo-compiler: the target environment may not
            # have it yet.
            lib = distribution("mojo-compiler").locate_file("modular/lib")
            for name in RUNTIME:
                shutil.copy(str(lib / f"{name}{LIBRARY_SUFFIX}"), module.parent)
            rpath = ORIGIN
        else:
            # Built outside the project, so a wheel doesn't replace an editable
            # install's module.
            self.build_dir = tempfile.TemporaryDirectory()
            module = Path(self.build_dir.name) / "bdd.abi3.so"
            build_data["force_include"][str(module)] = "bitpattern/bdd.abi3.so"
            # site-packages/modular/lib, where mojo-compiler (a dependency) puts the
            # runtime libraries.
            rpath = f"{ORIGIN}/../modular/lib"
            # The symbol table is most of the module's size. Editable installs keep the
            # symbols for debuggers and profilers.
            flags += STRIP_FLAGS
        # Linker arguments, in place of mojo's default: an rpath into this build
        # environment, which an isolated build deletes.
        shared_libs = f"-Xlinker,-rpath,-Xlinker,{rpath}"
        env = dict(os.environ, **BUILD_ENV, MODULAR_MOJO_MAX_SHARED_LIBS=shared_libs)
        source = root / "src" / "lib.mojo"
        # The build environment's mojo, which pip and uv put first on PATH.
        build = ["mojo", "build", source, "--emit", "shared-lib", "-o", module, *flags]
        subprocess.run(build, check=True, env=env)
        build_data["pure_python"] = False
        # The stable ABI, from CPython 3.11 on: one wheel per platform.
        build_data["tag"] = f"cp311-abi3-{PLATFORM_TAG}"

    def finalize(
        self, version: str, build_data: dict[str, Any], artifact_path: str
    ) -> None:
        if version != "editable":
            self.build_dir.cleanup()
