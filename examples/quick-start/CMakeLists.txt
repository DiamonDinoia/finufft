# The quick-start recipes, run from a normal build:
#
#   cmake --preset dev -DFINUFFT_BUILD_QUICKSTART=ON
#   ctest --preset dev -L quickstart
#
# Each recipe is a standalone project that builds FINUFFT its own way, so none of
# them can be add_subdirectory'd here. ctest's --build-and-test drives a separate
# configure, build and run for each, which is what a user does. That also means
# each test rebuilds FINUFFT, hence the option and the generous timeout.
#
# The three below are the routes that need no install. find_package and the
# Makefile consume an installed prefix, and tools/ci/install-test.sh stages one
# and runs those two there.
#
# The backend and the linkage come from this build, so the recipes are tested in
# the configuration you are already in rather than in a fixed one.
set(quickstart_common
    -DCMAKE_BUILD_TYPE=$<CONFIG>
    -DFINUFFT_USE_DUCC0=${FINUFFT_USE_DUCC0}
    -DFINUFFT_STATIC_LINKING=${FINUFFT_STATIC_LINKING}
    -DFINUFFT_USE_OPENMP=${FINUFFT_USE_OPENMP}
)

foreach(route IN ITEMS fetchcontent cpm subdirectory)
    if(route STREQUAL "fetchcontent")
        # Each tool's own documented redirect, so the recipe under test is the
        # published one and not a variant of it.
        set(quickstart_redirect -DFETCHCONTENT_SOURCE_DIR_FINUFFT=${PROJECT_SOURCE_DIR})
    elseif(route STREQUAL "cpm")
        set(quickstart_redirect -DFINUFFT_SOURCE_DIR=${PROJECT_SOURCE_DIR} -DCPM_finufft_SOURCE=${PROJECT_SOURCE_DIR})
    else()
        set(quickstart_redirect -DFINUFFT_SOURCE_DIR=${PROJECT_SOURCE_DIR})
    endif()
    add_test(
        NAME quickstart_${route}
        COMMAND
            ${CMAKE_CTEST_COMMAND} --build-config $<CONFIG> --build-and-test ${CMAKE_CURRENT_SOURCE_DIR}/${route}
            ${CMAKE_CURRENT_BINARY_DIR}/${route} --build-generator ${CMAKE_GENERATOR} --build-options
            ${quickstart_common} ${quickstart_redirect} --test-command
            # Quoted: CMAKE_CONFIGURATION_TYPES is a list, and an unquoted
            # argument splits on its semicolons before the genex is evaluated,
            # leaving three garbage arguments and no path. A multi-config
            # generator puts the executable under a config directory, a
            # single-config one beside the build directory.
            "${CMAKE_CURRENT_BINARY_DIR}/${route}/$<$<BOOL:${CMAKE_CONFIGURATION_TYPES}>:$<CONFIG>/>app"
    )
    # Not a unit test: one of these is a full FINUFFT build, so it is labelled to
    # be selected on purpose and given room to finish.
    set_tests_properties(quickstart_${route} PROPERTIES LABELS quickstart TIMEOUT 3600)
endforeach()
