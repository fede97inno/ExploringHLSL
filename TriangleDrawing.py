import glfw
from compushady import Texture2D, Swapchain, Compute, Buffer, HEAP_UPLOAD, HEAP_DEFAULT, HEAP_READBACK
from compushady.formats import B8G8R8A8_UNORM
from compushady.shaders import hlsl
from compushady.formats import R32G32B32A32_FLOAT, R32_UINT, R32G32B32_FLOAT
import struct

import math
import numpy as np
import compushady.config
compushady.config.set_debug(True)

def open_shader(path):
    with open(path, 'r') as file:
        hlsl_code = file.read()

    return hlsl_code

def identity_matrix():
    return np.array((
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    ), dtype= np.float32)

def translation_matrix(x, y, z):
    return np.array(
        (
            (1, 0, 0, x),
            (0, 1, 0, y),
            (0, 0, 1, z),
            (0, 0, 0, 1),
        ),
        dtype=np.float32,
    )

def rotation_matrix_y(angle):
    return np.array((
        (np.cos(angle), 0, np.sin(angle),   0),
        (0,             1, 0,               0),
        (-np.sin(angle),0, np.cos(angle),   0),
        (0,             0, 0,               1),
    ), dtype=np.float32
    )

def perspective_matrix_fov(fov, aspect_ratio, near, far):    #distances from the center
    top = near * np.tan(fov / 2)
    bottom = -top
    right = top * aspect_ratio     
    left = -right
    return perspective_matrix(left, right, top, bottom, near, far)

def perspective_matrix(left, right, top, bottom, near, far):
    return np.array((
        ((2 * near) / (right - left), 0, 0, (right+left) / (right-left)),
        (0, (2 * near) / (top-bottom), 0, (top+bottom)/(top-bottom)),
        (0, 0, -(far + near) / (far-near), -(2 * far * near)/(far-near)),
        (0, 0, -1, 0)
    ), dtype=np.float32
    )

triangles_info = []

stats_buffer = Buffer(64, HEAP_DEFAULT, format = R32_UINT)
stats_buffer_readback = Buffer(stats_buffer.size, HEAP_READBACK)

def add_triangle_buffer(x1, y1, z1, x2, y2, z2, x3, y3, z3, colorR, colorG, colorB):
    triangles_info.append((x1, y1, z1, x2, y2, z2, x3, y3, z3, colorR, colorG, colorB))
    pass

def upload_triangles(triangles):
    triangle_buffer = Buffer(len(triangles) * 12 * 4, HEAP_UPLOAD)
    for index, triangle in enumerate(triangles):
        triangle_buffer.upload(struct.pack("<12f", triangle[0], triangle[1], triangle[2], triangle[3], 
                                                   triangle[4], triangle[5], triangle[6], triangle[7],
                                                   triangle[8], triangle[9], triangle[10], triangle[11]
                                                   ), index * 12 * 4)
    
    fast_buffer = Buffer(len(triangles) * 12 * 4, HEAP_DEFAULT, format=R32G32B32_FLOAT)
    triangle_buffer.copy_to(fast_buffer)
    return fast_buffer

add_triangle_buffer(0, 0.8, 0, -0.8, -0.8, 0, 0.8, -0.8, 0, 0, 1, 0)
# add_triangle_buffer(-0.3, -0.3, 0.2, 0.2, 1, 1, 1, 1, 0)
# add_triangle_buffer(-0.3, -0.3, 0.2, 0.2, 1, 1, 1, 1, 0)

#new_fast_buffer = upload_rectangles(rectangles_info)
triangle_fast_buffer = upload_triangles(triangles_info)

shader = hlsl.compile(open_shader('shaders/triangle_shader.hlsl'))

clear_shader = hlsl.compile(open_shader("shaders/clear_shader.hlsl"))

configBuffer = Buffer(4 * 4 * 4 * 2 + 12, HEAP_UPLOAD)

perspective = perspective_matrix_fov(math.radians(90), 1, 0.01, 100)

glfw.init()
glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)

target = Texture2D(512,512, B8G8R8A8_UNORM)
compute = Compute(shader, uav=[target, stats_buffer], srv=[triangle_fast_buffer], cbv=[configBuffer])
clear_compute = Compute(clear_shader, uav=[target, stats_buffer])

window = glfw.create_window(target.width,target.height, "Hello", None, None)

swapchain = Swapchain(glfw.get_win32_window(window), B8G8R8A8_UNORM,2)

x,y = 0,0
camera_x, camera_y, camera_z= 0, 0, 0

def key_event(window,key,scancode,action,mods):
    global camera_x
    global camera_y
    global camera_z
    speed = 0.01

    if key == glfw.KEY_W and (action == glfw.PRESS or action == glfw.REPEAT):
        camera_y -= speed
    if key == glfw.KEY_S and (action == glfw.PRESS or action == glfw.REPEAT):
        camera_y += speed
    if key == glfw.KEY_A and (action == glfw.PRESS or action == glfw.REPEAT):
        camera_x -= speed
    if key == glfw.KEY_D and (action == glfw.PRESS or action == glfw.REPEAT):
        camera_x += speed
    if key == glfw.KEY_Q and (action == glfw.PRESS or action == glfw.REPEAT):
        camera_z -= speed
    if key == glfw.KEY_E and (action == glfw.PRESS or action == glfw.REPEAT):
        camera_z += speed



glfw.set_key_callback(window,key_event)
while not glfw.window_should_close(window):
    glfw.poll_events()

    x+=1
    y+=1
    configBuffer.upload(perspective.tobytes() + translation_matrix(camera_x, camera_y, camera_z).tobytes() + struct.pack("<ffI", x, y, len(triangles_info)))

    clear_compute.dispatch(target.width // 8, target.height // 8, 1)
    compute.dispatch(target.width //8, target.height // 8, 1)
    swapchain.present(target)
    stats_buffer.copy_to(stats_buffer_readback)
    print(struct.unpack("<I", stats_buffer_readback.readback(4)))