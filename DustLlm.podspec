# Xcode 26 beta workaround: duplicate .modulemap files in the SDK cause
# "redefinition of module" errors with explicit modules enabled.
xcode_major = begin
  m = `xcrun xcodebuild -version 2>/dev/null`.to_s.match(/Xcode (\d+)/)
  m ? m[1].to_i : 0
rescue
  0
end

Pod::Spec.new do |s|
  s.name = 'DustLlm'
  s.version = File.read(File.join(__dir__, 'VERSION')).strip
  s.summary = 'Standalone GGUF/llama.cpp inference and chat runtime for Dust.'
  s.license = { :type => 'Apache-2.0', :file => 'LICENSE' }
  s.homepage = 'https://github.com/rogelioRuiz/dust-llm-swift'
  s.author = 'Techxagon'
  s.source = { :git => 'https://github.com/rogelioRuiz/dust-llm-swift.git', :tag => s.version.to_s }

  s.source_files = [
    'Sources/DustLlm/**/*.swift',
    'native/llama.cpp/include/**/*.{h,hpp}',
    'native/llama.cpp/src/**/*.{c,cc,cpp,h,hpp,m,mm}',
    'native/llama.cpp/ggml/include/**/*.{h,hpp}',
    'native/llama.cpp/ggml/src/**/*.{c,cc,cpp,h,hpp,m,mm}',
    'native/llama.cpp/vendor/**/*.{h,hpp}',
    'native/llama.cpp/tools/mtmd/**/*.{c,cc,cpp,h,hpp}',
  ]
  s.exclude_files = [
    'native/llama.cpp/common/**/*',
    'native/llama.cpp/docs/**/*',
    'native/llama.cpp/examples/**/*',
    'native/llama.cpp/gguf-py/**/*',
    'native/llama.cpp/models/**/*',
    'native/llama.cpp/scripts/**/*',
    'native/llama.cpp/tests/**/*',
    'native/llama.cpp/tools/mtmd/deprecation-warning.cpp',
    'native/llama.cpp/tools/mtmd/mtmd-cli.cpp',
    'native/llama.cpp/ggml/src/ggml-cpu/arch/loongarch/**/*',
    'native/llama.cpp/ggml/src/ggml-cpu/arch/powerpc/**/*',
    'native/llama.cpp/ggml/src/ggml-cpu/arch/riscv/**/*',
    'native/llama.cpp/ggml/src/ggml-cpu/arch/s390/**/*',
    'native/llama.cpp/ggml/src/ggml-cpu/arch/wasm/**/*',
    'native/llama.cpp/ggml/src/ggml-cpu/arch/x86/**/*',
    'native/llama.cpp/ggml/src/ggml-cpu/spacemit/**/*',
    'native/llama.cpp/ggml/src/ggml-cpu/kleidiai/**/*',
  ]
  s.ios.deployment_target = '16.0'
  s.module_name = 'DustLlm'

  s.dependency 'DustCore'
  s.frameworks = ['Accelerate', 'Metal', 'MetalKit']
  xcconfig = {
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'HEADER_SEARCH_PATHS' => '$(inherited) $(PODS_TARGET_SRCROOT)/native/llama.cpp/include $(PODS_TARGET_SRCROOT)/native/llama.cpp/ggml/include $(PODS_TARGET_SRCROOT)/native/llama.cpp/src $(PODS_TARGET_SRCROOT)/native/llama.cpp/ggml/src $(PODS_TARGET_SRCROOT)/native/llama.cpp/vendor $(PODS_TARGET_SRCROOT)/native/llama.cpp/tools/mtmd',
    'GCC_PREPROCESSOR_DEFINITIONS' => '$(inherited) GGML_USE_METAL=1 GGML_METAL_EMBED_LIBRARY=1 GGML_VERSION=\"0.9.7\" GGML_COMMIT=\"b8189\"'
  }
  xcconfig['SWIFT_ENABLE_EXPLICIT_MODULES'] = 'NO' if xcode_major >= 26
  s.pod_target_xcconfig = xcconfig
  s.swift_version = '5.9'
end
