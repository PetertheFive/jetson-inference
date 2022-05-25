
file(GLOB posenetSources *.cpp)
file(GLOB posenetIncludes *.h )

find_package (OpenCV REQUIRED)
include_directories (${OpenCV_INCLUDE_DIRS})

cuda_add_executable(posenet ${posenetSources})
#target_link_libraries(posenet jetson-inference)
target_link_libraries(posenet jetson-inference ${OpenCV_LIBS})
install(TARGETS posenet DESTINATION bin)

