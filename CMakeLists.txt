cmake_minimum_required(VERSION 2.8.12)
project(Stereo-Image-to-PointCloud)

find_package( Pangolin REQUIRED )
find_package( OpenCV REQUIRED )
find_package( PCL REQUIRED )

add_executable(pointcloud src/main.cpp)
target_link_libraries(pointcloud ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(pcl_point src/pcl_main.cpp)
target_link_libraries(pcl_point ${OpenCV_LIBS} ${PCL_LIBRARIES})