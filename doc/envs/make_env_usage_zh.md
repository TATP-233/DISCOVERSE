# MuJoCo环境生成器 (make_env) 使用文档

## 1. 功能概述

`make_env`功能实现了动态组合机械臂和任务的功能，可以将不同的机械臂模型与任务场景组合成完整的仿真环境。这解决了当前项目中任务文件只能使用特定机械臂（如airbot_play）的限制，提供了更好的通用性和灵活性。

## 2. 主要特性

- **动态组合**：可以将任意机械臂与任意任务进行组合
- **路径转换**：自动将相对路径转换为绝对路径，确保MuJoCo能正确加载资源
- **XML操作**：提供完整的XML操作功能（获取XML对象、字符串、导出文件）
- **MuJoCo测试**：内置MuJoCo加载测试功能，验证生成的环境是否有效

## 3. 可用机械臂和任务

### 可用机械臂（robot_*.xml）
- `airbot_play` - AirBot Play机械臂
- `arx_l5` - ARX L5机械臂
- `arx_x5` - ARX X5机械臂
- `iiwa14` - KUKA iiwa14机械臂
- `panda` - Franka Panda机械臂
- `piper` - Piper机械臂
- `rm65` - Realman RM65机械臂
- `ur5e` - Universal Robots UR5e机械臂
- `xarm7` - xArm7机械臂

### 可用任务（tasks_airbot_play/*.xml）
- `block_bridge_place` - 积木桥放置任务
- `close_laptop` - 关闭笔记本电脑任务
- `cover_cup` - 盖杯子任务
- `open_drawer` - 开抽屉任务
- `pick_jujube` - 拾取枣子任务
- `place_block` - 放置积木任务
- `place_coffeecup` - 放置咖啡杯任务
- `place_jujube` - 放置枣子任务
- `place_jujube_coffeecup` - 放置枣子到咖啡杯任务
- `place_kiwi_fruit` - 放置猕猴桃任务
- `push_mouse` - 推鼠标任务
- `stack_block` - 堆叠积木任务
- `stack_two_colors_of_blocks` - 堆叠两种颜色积木任务

## 4. 基本用法

### 4.1 导入模块

```python
from discoverse.envs import make_env, list_available_robots, list_available_tasks
```

### 4.2 查看可用选项

```python
# 列出所有可用的机械臂
robots = list_available_robots()
print("可用机械臂:", robots)

# 列出所有可用的任务
tasks = list_available_tasks()
print("可用任务:", tasks)
```

### 4.3 创建环境

```python
# 创建环境：组合panda机械臂和stack_block任务
env = make_env("panda", "stack_block", "my_environment.xml")

# 获取XML字符串
xml_string = env.get_xml_string()

# 获取XML根元素（用于进一步处理）
xml_root = env.get_xml_root()
```

### 4.4 测试环境

```python
# 测试生成的环境是否能被MuJoCo加载
if env.test_mujoco_load():
    print("环境创建成功，可以被MuJoCo加载！")
else:
    print("环境有问题，无法被MuJoCo加载")
```

## 5. 高级用法

### 5.1 不导出文件直接使用

```python
# 不指定输出路径，只在内存中创建环境
env = make_env("ur5e", "cover_cup")

# 后续可以手动导出
env.export_xml("my_custom_path.xml")
```

### 5.2 批量创建环境

```python
robots = ["panda", "ur5e", "airbot_play"]
tasks = ["stack_block", "cover_cup", "place_block"]

for robot in robots:
    for task in tasks:
        try:
            env = make_env(robot, task, f"{robot}_{task}.xml")
            if env.test_mujoco_load():
                print(f"✓ {robot} + {task} 组合成功")
            else:
                print(f"✗ {robot} + {task} 组合失败")
        except FileNotFoundError as e:
            print(f"跳过 {robot} + {task}: {e}")
```

## 6. 实现原理

### 6.1 XML合并策略

1. **基础结构**：以机械臂XML为基础，复制其所有非worldbody元素
2. **worldbody合并**：
   - 首先添加机械臂的worldbody内容
   - 然后添加任务的worldbody内容（跳过机械臂相关的body）
3. **元素补充**：从任务XML复制其他缺失的元素（如actuator, sensor等）

### 6.2 路径处理

自动将XML中的相对路径转换为绝对路径：
- `mesh` 元素的 `file` 属性
- `texture` 元素的 `file` 属性  
- `include` 元素的 `file` 属性

### 6.3 文件结构要求

- 机械臂文件：`models/mjcf/robot_{robot_name}.xml`
- 任务文件：`models/mjcf/tasks_airbot_play/{task_name}.xml`
- 资源文件：基于`DISCOVERSE_ASSETS_DIR`的相对路径

## 7. 错误处理

### 7.1 常见错误

```python
try:
    env = make_env("nonexistent_robot", "stack_block")
except FileNotFoundError as e:
    print(f"机械臂文件不存在: {e}")

try:
    env = make_env("panda", "nonexistent_task")
except FileNotFoundError as e:
    print(f"任务文件不存在: {e}")
```

### 7.2 调试技巧

```python
# 检查生成的XML内容
env = make_env("panda", "stack_block")
xml_content = env.get_xml_string()

# 检查路径转换是否正确
from discoverse import DISCOVERSE_ASSETS_DIR
if DISCOVERSE_ASSETS_DIR in xml_content:
    print("路径转换正确")
else:
    print("路径转换可能有问题")
```

## 8. 示例：完整工作流程

```python
#!/usr/bin/env python3
"""
完整的make_env使用示例
"""

from discoverse.envs import make_env, list_available_robots, list_available_tasks

def main():
    # 1. 查看可用选项
    print("=== 可用选项 ===")
    robots = list_available_robots()
    tasks = list_available_tasks()
    print(f"机械臂 ({len(robots)}个): {robots}")
    print(f"任务 ({len(tasks)}个): {tasks}")
    
    # 2. 创建环境
    print("\n=== 创建环境 ===")
    robot_name = "panda"
    task_name = "stack_block"
    output_file = f"{robot_name}_{task_name}_environment.xml"
    
    try:
        env = make_env(robot_name, task_name, output_file)
        print(f"环境创建成功，XML文件已保存到: {output_file}")
        
        # 3. 验证环境
        print("\n=== 验证环境 ===")
        if env.test_mujoco_load():
            print("✓ 环境通过MuJoCo加载测试")
        else:
            print("✗ 环境无法被MuJoCo加载")
        
        # 4. 显示统计信息
        xml_string = env.get_xml_string()
        print(f"XML字符串长度: {len(xml_string)} 字符")
        
        # 5. 使用环境（示例）
        import mujoco
        model = mujoco.MjModel.from_xml_path(output_file)
        data = mujoco.MjData(model)
        print(f"模型包含 {model.nq} 个关节, {model.nbody} 个刚体")
        
    except Exception as e:
        print(f"创建环境失败: {e}")

if __name__ == "__main__":
    main()
```

## 9. 注意事项

1. **路径依赖**：确保`DISCOVERSE_ASSETS_DIR`正确设置，指向包含`models`目录的路径
2. **文件完整性**：确保机械臂和任务的所有依赖文件都存在
3. **命名冲突**：不同机械臂的body名称可能冲突，需要注意检查
4. **MuJoCo版本**：确保使用的MuJoCo版本支持生成的XML格式
5. **资源限制**：大型场景可能需要较多内存和计算资源

## 10. 扩展功能

### 10.1 自定义合并策略

可以通过修改`_merge_robot_and_task`函数来实现自定义的合并策略，例如：
- 调整机械臂在场景中的位置
- 修改物体的初始状态
- 添加额外的传感器或执行器

### 10.2 场景参数化

可以扩展`make_env`函数，支持场景参数化：
```python
def make_env_with_params(robot_name, task_name, robot_pos=(0, 0, 0.78), output_path=None):
    # 实现参数化的场景生成
    pass
```

这个功能为DISCOVERSE项目提供了强大的环境组合能力，使研究人员能够轻松测试不同机械臂在各种任务中的表现。 