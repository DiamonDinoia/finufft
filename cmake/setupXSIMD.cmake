CPMAddPackage(
    NAME
    xsimd
    GIT_REPOSITORY
    "https://github.com/xtensor-stack/xsimd.git"
    GIT_TAG
    ${XSIMD_VERSION}
    EXCLUDE_FROM_ALL
    YES
    SYSTEM
    YES
    GIT_SHALLOW
    NO
    OPTIONS
    "XSIMD_SKIP_INSTALL YES"
)

# finufft's Release flags enable -fassociative-math. Clang defines no macro for
# it, so xsimd's own detection leaves XSIMD_REASSOCIATIVE_MATH at 0 and compiles
# out the reassociation barriers that guard its SSE2 round/trunc/floor
# emulation. Those functions then return their input unchanged.
target_compile_definitions(
    xsimd
    INTERFACE
    $<$<CONFIG:Release,RelWithDebInfo>:XSIMD_REASSOCIATIVE_MATH=1>
)
