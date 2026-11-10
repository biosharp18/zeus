"""ZeUS rfdiffusion inference runtime."""


def _load_cuda_wheel_libraries():
    """Make the CUDA 11 wheel libraries visible to DGL's dynamic loader.

    DGL 0.6 links CUDA libraries by their standard SONAMEs. PyTorch 1.9's
    bundled libraries do not provide all of those names, and NVIDIA's wheel
    directories are not on the system loader path. Loading them here avoids
    an activation script or a machine-specific LD_LIBRARY_PATH. Environments
    without NVIDIA wheels can still supply these libraries through the OS.
    """
    import ctypes
    from importlib.util import find_spec
    from pathlib import Path

    if find_spec("nvidia") is None:
        return
    for package, library in [
        ("cuda_runtime", "libcudart.so.11.0"),
        ("cublas", "libcublas.so.11"),
        ("curand", "libcurand.so.10"),
        ("cusparse", "libcusparse.so.11"),
    ]:
        spec = find_spec("nvidia." + package)
        if spec is not None:
            directory = Path(next(iter(spec.submodule_search_locations)))
            ctypes.CDLL(str(directory / "lib" / library), mode=ctypes.RTLD_GLOBAL)


_load_cuda_wheel_libraries()
del _load_cuda_wheel_libraries
