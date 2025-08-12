if(FINUFFT_USE_CPU)
    install(TARGETS finufft LIBRARY DESTINATION finufft RUNTIME DESTINATION finufft)
    if(ENV{EDITABLE_REBUILD} STREQUAL "true")
        install(
            TARGETS finufft
            LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/finufft/finufft
            RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/finufft/finufft
        )
        set_target_properties(finufft PROPERTIES INSTALL_RPATH "$ORIGIN" BUILD_WITH_INSTALL_RPATH TRUE)
    endif()
endif()

if(FINUFFT_USE_CUDA)
    install(TARGETS cufinufft LIBRARY DESTINATION cufinufft RUNTIME DESTINATION cufinufft)
    if(ENV{EDITABLE_REBUILD} STREQUAL "true")
        install(
            TARGETS cufinufft
            LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/cufinufft/cufinufft
            RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/cufinufft/cufinufft
        )
        set_target_properties(cufinufft PROPERTIES INSTALL_RPATH "$ORIGIN" BUILD_WITH_INSTALL_RPATH TRUE)
    endif()
endif()

if(NOT SKBUILD)
    message(WARNING "This CMake file is meant to be executed via scikit-build-core.")
endif()
