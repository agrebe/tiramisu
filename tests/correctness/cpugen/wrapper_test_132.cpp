#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_132.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

int main(int, char **)
{
    Halide::Buffer<uint8_t> reference_buf0(SIZE0, SIZE0, "reference_buf0");
    init_buffer(reference_buf0, (uint8_t)2);

    for (int i = 1; i < SIZE0-1; i++)
	for (int j = 1; j < SIZE0-1; j++)
	    reference_buf0(j, i) = reference_buf0(j, i - 1) + reference_buf0(j - 1, i);

    Halide::Buffer<uint8_t> output_buf0(SIZE0, "output_buf0");
    init_buffer(output_buf0, (uint8_t)2);

    // Call the Tiramisu generated code
    tiramisu_generated_code(output_buf0.raw_buffer());

    compare_buffers(std::string(TEST_NAME_STR), output_buf0, reference_buf0);

    return 0;
}