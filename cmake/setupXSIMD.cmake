# Prefer a system package if available
find_package(xsimd CONFIG QUIET)

if(NOT xsimd_FOUND)
    CPMAddPackage(
        NAME
        xsimd
        GITHUB_REPOSITORY
        xtensor-stack/xsimd
        GIT_TAG
        ${XSIMD_VERSION}
    )
endif()

# Ensure a usable target exists; normalize to xsimd::xsimd if possible
if(TARGET xsimd::xsimd)
    # good
elseif(TARGET xsimd)
    # create a namespaced alias for convenience
    add_library(xsimd::xsimd INTERFACE IMPORTED)
    set_target_properties(xsimd::xsimd PROPERTIES INTERFACE_LINK_LIBRARIES xsimd)
else()
    # Header-only fallback target
    add_library(xsimd INTERFACE)
    target_include_directories(
        xsimd
        INTERFACE $<BUILD_INTERFACE:${xsimd_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
    )
    add_library(xsimd::xsimd ALIAS xsimd)
endif()
