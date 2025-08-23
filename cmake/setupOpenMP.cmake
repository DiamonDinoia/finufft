# cmake/SetupOpenMP.cmake
# Creates OpenMP::OpenMP_CXX (and OpenMP::OpenMP_C).
# Default: use system OpenMP (find_package(OpenMP)).
# If PREFER_MATLAB_IOMP5 is given, on macOS it will try MATLAB's libiomp5 and
# fall back to system OpenMP if not found.

include_guard(GLOBAL)

function(setup_omp)
    set(options REQUIRED PREFER_MATLAB_IOMP5)
    set(oneValueArgs)
    set(multiValueArgs LANGUAGES)
    cmake_parse_arguments(SETUP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT SETUP_LANGUAGES)
        set(SETUP_LANGUAGES C CXX)
    endif()

    set(_use_matlab_iomp5 OFF)
    if(SETUP_PREFER_MATLAB_IOMP5 AND APPLE)
        set(_use_matlab_iomp5 ON)
    endif()

    # Helper: ensure OpenMP:: targets exist using system OpenMP
    function(_setup_system_openmp _req)
        if(${_req})
            find_package(OpenMP REQUIRED)
        else()
            find_package(OpenMP)
        endif()
        foreach(lang IN LISTS SETUP_LANGUAGES)
            string(TOUPPER "${lang}" _LANG)
            set(_tgt "OpenMP::OpenMP_${_LANG}")
            if(TARGET "${_tgt}")
                continue()
            elseif(OpenMP_${_LANG}_FOUND)
                add_library("${_tgt}" INTERFACE IMPORTED)
                if(DEFINED OpenMP_${_LANG}_FLAGS)
                    separate_arguments(_omp_flags NATIVE_COMMAND "${OpenMP_${_LANG}_FLAGS}")
                    target_compile_options("${_tgt}" INTERFACE "${_omp_flags}")
                endif()
                if(DEFINED OpenMP_${_LANG}_LIBRARIES)
                    target_link_libraries("${_tgt}" INTERFACE ${OpenMP_${_LANG}_LIBRARIES})
                endif()
            elseif(${_req})
                message(FATAL_ERROR "setup_omp(): OpenMP ${_LANG} not found.")
            endif()
        endforeach()
    endfunction()

    if(_use_matlab_iomp5)
        if(NOT DEFINED Matlab_ROOT_DIR)
            message(STATUS "setup_omp(): Matlab_ROOT_DIR not set; falling back to system OpenMP.")
            _setup_system_openmp(${SETUP_REQUIRED})
            return()
        endif()

        # Try both Intel and Apple Silicon MATLAB layouts
        set(_iomp5_candidates
            "${Matlab_ROOT_DIR}/bin/maci64/libiomp5.dylib"
            "${Matlab_ROOT_DIR}/bin/maca64/libiomp5.dylib"
        )
        set(_iomp5 "")
        foreach(p IN LISTS _iomp5_candidates)
            if(EXISTS "${p}")
                set(_iomp5 "${p}")
                break()
            endif()
        endforeach()

        if(_iomp5)
            if(NOT TARGET SetupOMP::iomp5)
                add_library(SetupOMP::iomp5 SHARED IMPORTED GLOBAL)
                set_target_properties(SetupOMP::iomp5 PROPERTIES IMPORTED_LOCATION "${_iomp5}")
            endif()

            foreach(lang IN LISTS SETUP_LANGUAGES)
                string(TOUPPER "${lang}" _LANG)
                set(_tgt "OpenMP::OpenMP_${_LANG}")
                if(NOT TARGET "${_tgt}")
                    add_library("${_tgt}" INTERFACE IMPORTED)
                    if(_LANG STREQUAL "C" OR _LANG STREQUAL "CXX")
                        target_compile_options("${_tgt}" INTERFACE -Xpreprocessor -fopenmp)
                    endif()
                    target_link_libraries("${_tgt}" INTERFACE SetupOMP::iomp5)
                    target_link_options("${_tgt}" INTERFACE "-Wl,-rpath,@loader_path")
                endif()
            endforeach()
            message(STATUS "setup_omp(): Using MATLAB libiomp5 at ${_iomp5}")
        else()
            message(STATUS "setup_omp(): MATLAB libiomp5 not found; falling back to system OpenMP.")
            _setup_system_openmp(${SETUP_REQUIRED})
        endif()
    else()
        _setup_system_openmp(${SETUP_REQUIRED})
    endif()
endfunction()
