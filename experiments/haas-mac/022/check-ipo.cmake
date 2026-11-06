# This hook runs after the parent project has selected its C and C++ compilers.
function(check_compiler_ipo)
    # The probe's synthetic main function has no matching training profile.
    set(CMAKE_C_FLAGS "")
    set(CMAKE_CXX_FLAGS "")
    include(CheckIPOSupported)
    check_ipo_supported(LANGUAGES C CXX)
endfunction()

if(GGML_LTO OR CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    check_compiler_ipo()
endif()

if(CMAKE_C_FLAGS MATCHES "fprofile-instr-use=")
    # Diagnostic executables have different main functions from the trained worker.
    file(GLOB_RECURSE diagnostic_sources CONFIGURE_DEPENDS
         "${CMAKE_CURRENT_SOURCE_DIR}/checks/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/checks/*.cpp"
         "${CMAKE_CURRENT_SOURCE_DIR}/measure/*.cpp")
    set_property(SOURCE ${diagnostic_sources} APPEND PROPERTY COMPILE_OPTIONS -fno-profile-instr-use)
endif()
