get_filename_component(SOURCE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

vcpkg_check_features(
    OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        ducc0 FINUFFT_USE_DUCC0
        openmp FINUFFT_USE_OPENMP
)

string(COMPARE EQUAL "${VCPKG_LIBRARY_LINKAGE}" "static" FINUFFT_STATIC)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        ${FEATURE_OPTIONS}
        -DFINUFFT_STATIC_LINKING=${FINUFFT_STATIC}
        -DFINUFFT_BUILD_TESTS=OFF
        -DFINUFFT_BUILD_EXAMPLES=OFF
        -DFINUFFT_BUILD_PYTHON=OFF
        -DFINUFFT_ENABLE_INSTALL=ON
        -DFINUFFT_FFTW_LIBRARIES=DEFAULT
        -DFETCHCONTENT_FULLY_DISCONNECTED=OFF
    MAYBE_UNUSED_VARIABLES
        FINUFFT_FFTW_LIBRARIES
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME finufft CONFIG_PATH lib/cmake/finufft)
vcpkg_copy_pdbs()

file(
    REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/debug/include"
    "${CURRENT_PACKAGES_DIR}/debug/share"
    "${CURRENT_PACKAGES_DIR}/share/finufft/examples"
    "${CURRENT_PACKAGES_DIR}/share/licenses"
)

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
