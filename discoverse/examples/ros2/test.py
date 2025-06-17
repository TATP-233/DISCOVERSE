# test.py
import glfw
from OpenGL.GL import *  # 修正拼写错误
import numpy as np

def main():
    # 初始化GLFW
    if not glfw.init():
        print("GLFW初始化失败")
        return

    # 创建窗口
    window = glfw.create_window(800, 600, "OpenGL Window", None, None)
    if not window:
        print("窗口创建失败")
        glfw.terminate()
        return

    glfw.make_context_current(window)

    # 检查OpenGL版本
    version = glGetString(GL_VERSION)
    print(f"OpenGL版本: {version.decode()}")

    # 主循环
    while not glfw.window_should_close(window):
        # 处理输入
        glfw.poll_events()

        # 清除颜色缓冲区
        glClearColor(0.1, 0.2, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # 交换缓冲区
        glfw.swap_buffers(window)

    # 清理资源
    glfw.terminate()

if __name__ == "__main__":
    main()