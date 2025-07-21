# add specific source file to idefix build tree
add_idefix_source(cos.hpp)
add_idefix_source(analysis.cpp)
add_idefix_source(analysis.hpp)

replace_idefix_source(fluid/thermalDiffusion.cpp mythermalDiffusion.cpp)