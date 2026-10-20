"""Compiles bitpattern.bdd from src/lib.mojo, for wheels and editable installs."""

import os
import platform
import shutil
import subprocess
import sysconfig
import tempfile
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# The Mojo runtime library the module loads, and the libraries it loads.
RUNTIME = [
    "libKGENCompilerRTShared.so",
    "libAsyncRTRuntimeGlobals.so",
    "libMSupportGlobals.so",
]
# The module runs on any CPU of the architecture, not just the build machine's.
TARGET_CPUS = {"x86_64": "x86-64-v2"}


class MojoBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        root = Path(self.root)
        flags = ["-I", str(root / "src"), "-D", f"VERSION={self.metadata.version}"]
        # Source paths in the module are the project's, so a build doesn't depend on
        # where it ran.
        flags += ["-strip-file-prefix", f"{root}/"]
        if cpu := TARGET_CPUS.get(platform.machine()):
            flags += ["--target-cpu", cpu]
        if version == "editable":
            # The module stays in the project, away from site-packages, so it gets
            # copies of the runtime libraries.
            module = root / "bitpattern" / "bdd.abi3.so"
            lib = Path(sysconfig.get_path("platlib")) / "modular" / "lib"
            for name in RUNTIME:
                shutil.copy(lib / name, module.parent)
            rpath = "$ORIGIN"
        else:
            # Built outside the project, so a wheel doesn't replace an editable
            # install's module.
            self.build_dir = tempfile.TemporaryDirectory()
            module = Path(self.build_dir.name) / "bdd.abi3.so"
            build_data["force_include"][str(module)] = "bitpattern/bdd.abi3.so"
            # site-packages/modular/lib, where mojo-compiler (a dependency) puts the
            # runtime libraries.
            rpath = "$ORIGIN/../modular/lib"
        # Linker arguments, in place of mojo's default: an rpath into this build
        # environment, which an isolated build deletes.
        shared_libs = f"-Xlinker,-rpath,-Xlinker,{rpath}"
        env = dict(os.environ, MODULAR_MOJO_MAX_SHARED_LIBS=shared_libs)
        mojo = Path(sysconfig.get_path("scripts")) / "mojo"
        source = root / "src" / "lib.mojo"
        build = [mojo, "build", source, "--emit", "shared-lib", "-o", module, *flags]
        subprocess.run(build, check=True, env=env)
        build_data["pure_python"] = False
        # The stable ABI, from CPython 3.11 on: one wheel per platform.
        plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        build_data["tag"] = f"cp311-abi3-{plat}"

    def finalize(
        self, version: str, build_data: dict[str, Any], artifact_path: str
    ) -> None:
        if version != "editable":
            self.build_dir.cleanup()
