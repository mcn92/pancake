FROM ann-benchmarks

ARG PANCAKE_REV=865e76f

RUN apt-get update \
    && apt-get install -y --no-install-recommends git g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pybind11

RUN git clone https://github.com/mcn92/pancake.git /opt/pancake \
    && cd /opt/pancake \
    && git checkout "${PANCAKE_REV}" \
    && c++ -O3 -DNDEBUG -std=c++17 -shared -fPIC \
       -DPANCAKE_ENABLE_AVX2_SIMD -mavx2 -mfma \
       $(python3 -m pybind11 --includes) \
       -Isrc benchmarks/vibe/pancake_py.cpp \
       -o "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["platlib"] + "/pancake_py" + sysconfig.get_config_var("EXT_SUFFIX"))')"

RUN python3 -c "import pancake_py; assert hasattr(pancake_py, 'PancakeIndex')"
