# cmake needs this line
cmake_minimum_required(VERSION 3.13)

# Disable in-source builds to prevent source tree corruption.
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "
            FATAL: In-source builds are not allowed.
                You should create a separate directory for build files.
            ")
endif()

# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Define project name
project(The-Outex-TC12-Classification)

find_package(OpenCV REQUIRED)


# Declare the executable target built from your sources
add_executable(Texture_Classifier main.cpp)

# Link your application with OpenCV libraries
target_link_libraries(Texture_Classifier LINK_PRIVATE ${OpenCV_LIBS})
