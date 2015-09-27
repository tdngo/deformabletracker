# Build the realtime deformable surface 3D tracking and reconstruction
#
# Author: Ngo Tien Dat, datngotien@gmail.com
# 25 Sep 2014

cmake_minimum_required (VERSION 2.6)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)

set(LIB_NAME deformable)

# Find opencv
find_package( OpenCV REQUIRED )

# Find armadillo
find_library( armadillo NAMES "armadillo" PATHS /opt/local/lib/ )
message(STATUS "Found armadillo in ${armadillo}")

# Find ferns
find_library( ferns NAMES "ferns" PATHS "${PROJECT_SOURCE_DIR}/3rdparty/ferns/" )
message(STATUS "Found ferns in ${ferns}")

# Find timer
find_library( timer NAMES "timer" PATHS "${PROJECT_SOURCE_DIR}/3rdparty/timer/" )
message(STATUS "Found timer in ${timer}")

# Inlude directories
include_directories( ${OpenCV_INCLUDE_DIRS} )           # for OpenCV
include_directories( "${PROJECT_SOURCE_DIR}/src" )
include_directories( "${PROJECT_SOURCE_DIR}/3rdparty" ) # for ferns and timer
include_directories( "/opt/local/include/" )            # for armadillo

set(SOURCE_FILES 
    ./src/Camera.cpp
    ./src/Linear/EqualConstrFunction.cpp
    ./src/Linear/EqualConstrOptimize.cpp
    ./src/Linear/IneqConstrFunction.cpp
    ./src/Linear/IneqConstrOptimize.cpp
    ./src/Linear/LinearAlgebraUtils.cpp
    ./src/Linear/ObjectiveFunction.cpp
    ./src/MatchTracker/InlierMatchTracker.cpp
    ./src/MatchTracker/LKPointTracker.cpp
    ./src/Mesh/LaplacianMesh.cpp
    ./src/Mesh/TriangleMesh.cpp
    ./src/PointMatching/FernKeypointMatcher3D2D.cpp
    ./src/PointMatching/KeypointMatcher3D2D.cpp
    ./src/RealtimeDemo/RealtimeDemo.cpp
    ./src/Reconstruction.cpp
    ./src/Utils/DUtils.cpp
    ./src/Utils/Visualization.cpp        
)

set( LIBS ${armadillo} ${ferns} ${timer} ${OpenCV_LIBS} )

# Build library
add_library( ${LIB_NAME} SHARED ${SOURCE_FILES} )
target_link_libraries( ${LIB_NAME} ${LIBS})

# Build deformabletracker
set(APP_NAME deformabletracker)
add_executable( ${APP_NAME} ./src/test/test_deformable_realtime.cpp )
target_link_libraries( ${APP_NAME} ${LIB_NAME} ${LIBS} )



