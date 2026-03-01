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
  ]
  s.exclude_files = [
    'native/llama.cpp/common/**/*',
    'native/llama.cpp/docs/**/*',
    'native/llama.cpp/examples/**/*',
    'native/llama.cpp/gguf-py/**/*',
    'native/llama.cpp/models/**/*',
    'native/llama.cpp/scripts/**/*',
    'native/llama.cpp/tests/**/*',
    'native/llama.cpp/tools/**/*',
  ]
  s.ios.deployment_target = '16.0'
  s.module_name = 'DustLlm'

  s.dependency 'DustCore'
  s.frameworks = ['Accelerate', 'Metal', 'MetalKit']
  s.pod_target_xcconfig = {
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'HEADER_SEARCH_PATHS' => '$(inherited) $(PODS_TARGET_SRCROOT)/native/llama.cpp/include $(PODS_TARGET_SRCROOT)/native/llama.cpp/ggml/include $(PODS_TARGET_SRCROOT)/native/llama.cpp/src $(PODS_TARGET_SRCROOT)/native/llama.cpp/ggml/src $(PODS_TARGET_SRCROOT)/native/llama.cpp/vendor',
    'GCC_PREPROCESSOR_DEFINITIONS' => '$(inherited) GGML_USE_METAL=1 GGML_METAL_EMBED_LIBRARY=1'
  }
  s.swift_version = '5.9'
end
