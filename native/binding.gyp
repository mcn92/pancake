{
  "targets": [
    {
      "target_name": "pancake_native",
      "sources": ["pancake_napi.cpp"],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "../src"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "cflags!": ["-fno-exceptions"],
      "cflags_cc!": ["-fno-exceptions"],
      "cflags_cc": [
        "-O3",
        "-std=c++17",
        "-ffast-math",
        "-ftree-vectorize",
        "-fno-rtti",
        "-march=native",
        "-mavx2",
        "-msse2",
        "-DPANCAKE_ENABLE_AVX2_SIMD",
        "-DPANCAKE_ENABLE_SSE2_SIMD"
      ],
      "conditions": [
        ["OS=='mac'", {
          "xcode_settings": {
            "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
            "CLANG_CXX_LANGUAGE_STANDARD": "c++17",
            "OTHER_CPLUSPLUSFLAGS": [
              "-O3", "-ffast-math", "-fvectorize", "-fslp-vectorize",
              "-fno-rtti", "-mavx2",
              "-DPANCAKE_ENABLE_AVX2_SIMD", "-DPANCAKE_ENABLE_SSE2_SIMD"
            ]
          }
        }],
        ["OS=='win'", {
          "msvs_settings": {
            "VCCLCompilerTool": {
              "ExceptionHandling": 1,
              "AdditionalOptions": ["/O2", "/std:c++17", "/arch:AVX2", "/DPANCAKE_ENABLE_AVX2_SIMD", "/DPANCAKE_ENABLE_SSE2_SIMD"]
            }
          }
        }]
      ],
      "defines": ["NAPI_DISABLE_CPP_EXCEPTIONS"]
    }
  ]
}
