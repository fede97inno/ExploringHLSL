import glfw 
from compushady import Texture2D, Swapchain, Compute, Buffer, HEAP_UPLOAD, HEAP_DEFAULT, HEAP_READBACK
from compushady.formats import B8G8R8A8_UNORM, R32G32B32A32_FLOAT, R32_UINT
from compushady.shaders import hlsl
import struct
import compushady.config

#rectangle_buffer = Buffer(8 * 4 * 4 + 4 * 4, HEAP_UPLOAD) # 8 rectangles -> we need 16 bytes for every rect

def key_event(window,key,scancode,action,mods):
    global camera_x, camera_y
    speed = 20

    if key == glfw.KEY_W and (action == glfw.PRESS or action == glfw.REPEAT):
        camera_y -= speed
    if key == glfw.KEY_S and (action == glfw.PRESS or action == glfw.REPEAT):
        camera_y += speed
    if key == glfw.KEY_A and (action == glfw.PRESS or action == glfw.REPEAT):
        camera_x -= speed
    if key == glfw.KEY_D and (action == glfw.PRESS or action == glfw.REPEAT):
        camera_x += speed

rectangles = []
stats_buffer = Buffer(64, HEAP_DEFAULT, R32_UINT)
stats_readback = Buffer(stats_buffer.size, HEAP_READBACK)

def add_rectangle_to_buffer(x, y, w, h):
    rectangles.append((x,y,w,h))
    pass

def upload_rectangles(rectangles, colorR, colorG, colorB):
    rectangle_buffer = Buffer(len(rectangles) * 4 * 4 + 4 * 4, HEAP_UPLOAD)
    rectangle_buffer.upload(struct.pack("<Ifff", len(rectangles), colorR, colorG, colorB))
    for index, rectangle in enumerate(rectangles):     # return a tuple where first element is index second element is rectangle
        rectangle_buffer.upload(struct.pack("<ffff", rectangle[0], rectangle[1], rectangle[2], rectangle[3]), index * 4 * 4 + 4 * 4)
    
    fast_buffer = Buffer(len(rectangles) * 4 * 4 + 4 * 4, HEAP_DEFAULT, format=R32G32B32A32_FLOAT)
    rectangle_buffer.copy_to(fast_buffer)

    return fast_buffer

add_rectangle_to_buffer(0, 0, 10, 10)
add_rectangle_to_buffer(10, 0, 10, 10)
add_rectangle_to_buffer(0, 20, 10, 10)
add_rectangle_to_buffer(0, 40, 10, 10)
add_rectangle_to_buffer(124, 32, 10, 10)
add_rectangle_to_buffer(123, 35, 114, 10)
add_rectangle_to_buffer(32, 0, 89, 10)
add_rectangle_to_buffer(0, 53, 300, 10)

new_fast_buffer = upload_rectangles(rectangles, 0, 0, 1)

config_buffer = Buffer(8 + 8, HEAP_UPLOAD)

compushady.config.set_debug(True)

shader = hlsl.compile(
"""
RWTexture2D<float4> Target;     // register u0
Buffer<float4> RectangleBuffer;
RWBuffer<uint> Stats;

struct Config
{
    float DeltaX;
    float DeltaY;
    float2 Camera;
};

ConstantBuffer<Config> ConfigBuffer;

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadId)
{
    for(int index = 1; index < asuint(RectangleBuffer[0].r) ; index++)      // Reinterpret float RectangleBuffer[0].r into an uint
    {
        float4 rectangle = RectangleBuffer[index];
        rectangle.xy += ConfigBuffer.Camera;

        if (tid.x >= rectangle.x && tid.x < rectangle.x + rectangle.z && tid.y >= rectangle.y && tid.y < rectangle.y + rectangle.w)
        {
            Target[tid.xy] = float4(RectangleBuffer[0].yzw,1);
            InterlockedAdd(Stats[0], 1);
        }
    }
}
""")

# Atomic operation in hlsl are given by the language you can only use them with int types -> interlocked operation

clear_shader = hlsl.compile(
"""
RWTexture2D<float4> Target;
RWBuffer<uint> Stats;

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadId)
{
    Target[tid.xy] = float4(1, 0, 0, 1);
    Stats[0] = 0;
}
""")

glfw.init()

glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)  # we don't want opengl api integrated, without no_api is there by default

# 0 -> 0 255 -> 1
target = Texture2D(512, 512, B8G8R8A8_UNORM)    # B8G8R8A8_UNORM -> used for texture seen on windows, classic swap chain type || the construct that permits to swap from front and back buffer is call swapchain

compute = Compute(shader, uav=[target, stats_buffer], srv=[new_fast_buffer], cbv=[config_buffer])
clear_compute = Compute(clear_shader, uav=[target, stats_buffer])

window = glfw.create_window(target.width, target.height, "Hello", None, None)

swapchain = Swapchain(glfw.get_win32_window(window), B8G8R8A8_UNORM, 2)     # window ptr, texture type format, number of texture


glfw.set_key_callback(window,key_event)

delta_x, delta_y = 0, 0
camera_x, camera_y = 0, 0

while not glfw.window_should_close(window):
    glfw.poll_events()
    #delta_x += 1
    #delta_y += 1
    config_buffer.upload(struct.pack("<ffff", delta_x, delta_y, camera_x, camera_y))
    clear_compute.dispatch(target.width // 8, target.height // 8, 1)
    compute.dispatch(target.width // 8, target.height // 8, 1)      # // int python division
    swapchain.present(target) # texture to backbuffer to frontbuffer

    stats_buffer.copy_to(stats_readback)
    print(struct.unpack("<I", stats_readback.readback(4)))