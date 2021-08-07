find_package(Pangolin REQUIRED)
find_package( OpenCV REQUIRED )

add_executable(pointcloud src/main.cpp)
target_link_libraries(pointcloud ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
