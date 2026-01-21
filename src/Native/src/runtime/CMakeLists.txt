cmake_minimum_required (VERSION 3.13)

set(SRCS buffer.cpp
     copy.cpp
		 datatypes.cpp
		 error.cpp
     device_buffer.cpp
     host_allocator.cpp
		 host_buffer.cpp
		 host_runtime_tensor.cpp
     interpreter.cpp
     llm_interpreter.cpp
     runtime_section_context.cpp
     runtime_loader.cpp
     runtime_module.cpp
		 runtime_function.cpp
		 section.cpp
		 type_serializer.cpp
		 tensor_serializer.cpp
		 runtime_tensor.cpp
     dump_manager.cpp)

target_sources(nncaseruntime PRIVATE ${SRCS})
if (NOT (WIN32 OR APPLE))
  target_link_libraries(nncaseruntime PRIVATE dl pthread)
endif()

add_subdirectory(cpu)
add_subdirectory(cuda)
