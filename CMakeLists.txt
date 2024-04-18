project(FA_FWD CUDA)

set(CUTLASS_PATH C:\\code\\cutlass\\include)

include_directories(${CUTLASS_PATH} ${PROJECT_SOURCE_DIR})

aux_source_directory(${PROJECT_SOURCE_DIR}/source src)

add_executable(fa_fwd ${src})

add_definitions(-std=c++17)

target_link_libraries(fa_fwd cudart_static)
