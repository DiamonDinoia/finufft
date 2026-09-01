set(quickstart_common
    -DCMAKE_BUILD_TYPE=$<CONFIG>
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DFINUFFT_USE_DUCC0=${FINUFFT_USE_DUCC0}
    -DFINUFFT_STATIC_LINKING=${FINUFFT_STATIC_LINKING}
    -DFINUFFT_USE_OPENMP=${FINUFFT_USE_OPENMP}
)

foreach(route IN ITEMS fetchcontent cpm subdirectory)
    if(route STREQUAL "fetchcontent")
        set(quickstart_redirect -DFETCHCONTENT_SOURCE_DIR_FINUFFT=${PROJECT_SOURCE_DIR})
    elseif(route STREQUAL "cpm")
        set(quickstart_redirect -DCPM_finufft_SOURCE=${PROJECT_SOURCE_DIR})
    else()
        set(quickstart_redirect -DFINUFFT_SOURCE_DIR=${PROJECT_SOURCE_DIR})
    endif()
    add_test(
        NAME quickstart_${route}
        COMMAND
            ${CMAKE_CTEST_COMMAND} --build-config $<CONFIG> --build-and-test ${CMAKE_CURRENT_SOURCE_DIR}/${route}
            ${CMAKE_CURRENT_BINARY_DIR}/${route} --build-generator ${CMAKE_GENERATOR} --build-options
            ${quickstart_common} ${quickstart_redirect} --test-command
            "${CMAKE_CURRENT_BINARY_DIR}/${route}/$<$<BOOL:${CMAKE_CONFIGURATION_TYPES}>:$<CONFIG>/>app"
    )
    set_tests_properties(quickstart_${route} PROPERTIES LABELS quickstart TIMEOUT 3600)
endforeach()
