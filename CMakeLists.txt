# 1.声明要求最低的cmake版本
cmake_minimum_required(VERSION 2.8)
# set(CMAKE_THREAD_LIBS_INIT "-lpthread")
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
# set(CMAKE_HAVE_THREADS_LIBRARY 1)
# set(CMAKE_USE_WIN32_THREADS_INIT 0)
# set(CMAKE_USE_PTHREADS_INIT 1)
# set(THREADS_PREFER_PTHREAD_FLAG ON)

# 2.添加c++标准支持
set( CMAKE_CXX_FLAGS "-std=c++11 ")

# 3.声明一个cmake工程
PROJECT(VisionProj)
#MESSAGE(STATUS "Project: SERVER") #打印相关信息

# 是应用命令查找opencv库
find_package(OpenCV REQUIRED) #寻找opencv
find_package(CUDA REQUIRED) # 寻找CUDA

# message(STATUS "OpenCV library status:")
# message(STATUS "    version: ${OpenCV_VERSION}")
# message(STATUS "    libraries: ${OpenCV_LIBS}")
# message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# 4.头文件
include_directories(
    ${OpenCV_INCLUE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/inc
    ) #添加opencv头文件

# 5.通过设定SRC变量，将源代码路径都给SRC，如果有多个，可以直接在后面添加
set(SRC
    ${PROJECT_SOURCE_DIR}/src/main.cpp
    ${PROJECT_SOURCE_DIR}/src/GetImg.cpp
    ${PROJECT_SOURCE_DIR}/src/ProcImg.cpp
    ${PROJECT_SOURCE_DIR}/src/inference.cpp
    ${PROJECT_SOURCE_DIR}/src/logger.cpp
    )

# 6.创建共享库/静态库
# 设置路径（下面生成共享库路径）
# 即生成的共享库在工程文件夹下的lib文件夹中
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# 创建共享库（把工程内的cpp文件都创建成共享库文件，方便通过头文件调用）
# ${PROJECT_NAME}是生成的库名，表示生成的共享库文件就叫做lib工程名.so
# 也可以专门写cmakelists来编译一个没有主函数的程序来生成共享库，供其它程序使用
# SHARED为生成动态库，STATIC为生成静态库
# add_library( SHARED ${SRC})
# set(LIBGETIMG_SRC ${PROJECT_SOURCE_DIR}/src/GetImg.cpp)   # GetImg
# add_library(getimg SHARED ${LIBGETIMG_SRC})

# set(LIBCALIBRATECAMERA_SRC ${PROJECT_SOURCE_DIR}/src/CalibrateCamera.cpp)   # CalibrateCamera
# add_library(calibratecamera SHARED ${LIBCALIBRATECAMERA_SRC})

# 7. 链接库文件
# 把刚刚生成的${LIB_NAME}库和所需的其它库链接起来
# 如果需要链接其他的动态库，-l后接去除lib前缀和.so后缀的名称，以链接
# libpthread.so 为例,-lpthread
#target_link_libraries(${LIB_NAME} pthread dl)
link_directories(${PROJECT_SOURCE_DIR}/lib /usr/local/cuda/lib64)
set(LIBNAME gxiapi cudart cublas nvinfer)

# 8. 编译主函数，生成可执行文件
# 先设置路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# 可执行文件生成
add_executable(${PROJECT_NAME} ${SRC})

# 这个可执行文件所需的库（一般就是刚刚生成的工程的库）
#target_link_libraries(${PROJECT_NAME} pthread dl ${LIB_NAME})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${LIBNAME})