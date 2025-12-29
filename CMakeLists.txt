cmake_minimum_required(VERSION 4.1)
project(MininumCommandBufferPool)

set(CMAKE_CXX_STANDARD 20)

# Require Vulkan and GLFW
find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

add_executable(main
        main.cpp
)

# Include dirs
target_include_directories(main PRIVATE ${Vulkan_INCLUDE_DIRS})

# Link Vulkan and GLFW;

target_link_libraries(main PRIVATE Vulkan::Vulkan glfw)

# Use C++20
target_compile_features(main PRIVATE cxx_std_20)

# Enable MSVC warnings sensibly
if (MSVC)
    target_compile_options(main PRIVATE /W4)
else()
    target_compile_options(main PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
endif()
