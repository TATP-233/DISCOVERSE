import glfw
from OpenGL.GL import *
import sys

def main():
    if not glfw.init():
        print("Failed to initialize GLFW")
        sys.exit(1)

    window = glfw.create_window(640, 480, "OpenGL Test", None, None)
    if not window:
        print("Failed to create GLFW window")
        glfw.terminate()
        sys.exit(1)

    glfw.make_context_current(window)

    version = glGetString(GL_VERSION).decode()
    renderer = glGetString(GL_RENDERER).decode()
    print("OpenGL version:", version)
    print("Renderer:", renderer)

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
