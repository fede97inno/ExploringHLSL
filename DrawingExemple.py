# 8 int from 0 to 7 we want to double this values in the fastest way

# 4 bytes * 8 elements = 32 bytes

from compushady import Buffer, HEAP_UPLOAD, HEAP_DEFAULT, HEAP_READBACK, Compute
import struct
import compushady.config
from compushady.shaders import hlsl
from compushady.formats import R32_UINT

# RGBA pixel format RSize_Type

compushady.config.set_debug(True)

# With shaders we can do operation in GPU, we can use parallel threading

shader = hlsl.compile(
"""
RWBuffer<uint> FastBuffer;      // : register(u0)
[numthreads(8,1,1)]
void main(uint tid : SV_DispatchThreadId)
{
    FastBuffer[tid.x] *= 2;
}                      
"""
)

# in shader there are the opcodes

# HEAP_DEFAULT  -> GPU memory, access only form GPU
# HEAP_UPLOAD   -> CPU memory, write access from CPU, read access from GPU
# HEAP_READBACK -> CPU memory, write access from GPU, read access from CPU

source_buffer = Buffer(4 * 8, HEAP_UPLOAD)
readback_buffer = Buffer(4 * 8, HEAP_READBACK)
fast_buffer = Buffer(4 * 8, HEAP_DEFAULT, format=R32_UINT)  # every elements you ll read 32 bits every time. You need to specify how much a element is big, this R32_UINT is one of the Pixel formats everything is RGBA (for float4 is RGBA in This case we can use the R)

data = struct.pack("<IIIIIIII", 0, 1, 2, 3, 4, 5, 6, 7)

source_buffer.upload(data)              # data is on CPU
source_buffer.copy_to(fast_buffer)      # data is now on GPU

compute = Compute(shader, uav=[fast_buffer])
compute.dispatch(1, 1, 1)   # how many times execute the threads block (8,1,1) * (1,1,1) = 8 * 1 one time the eight threads

fast_buffer.copy_to(readback_buffer)    # result data is now on CPU

print(readback_buffer.readback())
print(struct.unpack("<8I", readback_buffer.readback()))