CPMAddPackage(
    NAME Callgraph
    GIT_REPOSITORY "git@github.com:DiamonDinoia/callgraph.git"
    GIT_TAG "main"
    EXCLUDE_FROM_ALL Yes
)

if(NOT INUFFT_STATIC_LINKING)
    set_property(TARGET callgraph PROPERTY POSITION_INDEPENDENT_CODE ON)
endif ()