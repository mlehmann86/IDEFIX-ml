# add specific source file to idefix build tree
add_idefix_source(cos.hpp)
add_idefix_source(analysis.cpp)
add_idefix_source(analysis.hpp)

replace_idefix_source(fluid/thermalDiffusion.cpp mythermalDiffusion.cpp)

# need GSL for eigenvalue solver
find_package(GSL REQUIRED)

# Check if GSL was found
if(GSL_FOUND)
    message(STATUS "GSL found: ${GSL_LIBRARIES}")

    # Use GSL_CONFIG_EXECUTABLE to get the GSL compile flags
    execute_process(
        COMMAND ${GSL_CONFIG_EXECUTABLE} --cflags
        OUTPUT_VARIABLE GSL_CFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Use GSL_CONFIG_EXECUTABLE to get the GSL library flags
    execute_process(
        COMMAND ${GSL_CONFIG_EXECUTABLE} --libs
        OUTPUT_VARIABLE GSL_LIBS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Print GSL CFLAGS and GSL LIBS for debugging
    message(STATUS "GSL CFLAGS: ${GSL_CFLAGS}")
    message(STATUS "GSL LIBS: ${GSL_LIBS}")

    target_link_libraries(idefix GSL::gsl GSL::gslcblas)
else()
    message(FATAL_ERROR "GSL not found!")
endif()
