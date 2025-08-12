function(copy_dll source_target destination_target)
    if(NOT WIN32)
        return()
    endif()
    add_custom_command(
        TARGET ${destination_target}
        POST_BUILD
        COMMAND
            ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:${source_target}>
            $<TARGET_FILE_DIR:${destination_target}>/$<TARGET_FILE_NAME:${source_target}>
        COMMENT "Copying $<TARGET_FILE_NAME:${source_target}> next to $<TARGET_FILE_NAME:${destination_target}>"
        VERBATIM
    )
endfunction()

include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERROR)
if(LTO_SUPPORTED)
    set(FINUFFT_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    set(FINUFFT_INTERPROCEDURAL_OPTIMIZATION FALSE)
endif()

function(detect_cuda_architecture)
    find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)
    if(NVIDIA_SMI_EXECUTABLE)
        execute_process(
            COMMAND ${NVIDIA_SMI_EXECUTABLE} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE compute_cap
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(compute_cap MATCHES "^[0-9]+\\.[0-9]+$")
            string(REPLACE "." "" arch "${compute_cap}")
            set(CMAKE_CUDA_ARCHITECTURES ${arch} PARENT_SCOPE)
        else()
            message(FATAL_ERROR "Failed to parse compute capability: '${compute_cap}'")
        endif()
    else()
        message(FATAL_ERROR "nvidia-smi not found; set -DCMAKE_CUDA_ARCHITECTURES=..")
    endif()
endfunction()
