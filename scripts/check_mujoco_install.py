import mujoco

xml_string = """
<mujoco>
    <worldbody>
        <geom name="floor" type="plane" size="0 0 0.1" rgba="0.8 0.9 0.8 1"/>
    </worldbody>
</mujoco>
"""

mj_model = mujoco.MjModel.from_xml_string(xml_string)
mj_data = mujoco.MjData(mj_model)
mujoco.mj_forward(mj_model, mj_data)

mujoco.mj_step(mj_model, mj_data)

renderer = mujoco.Renderer(mj_model)
renderer.update_scene(mj_data)

img = renderer.render()

print(img.shape, img.dtype)
print(img.max(), img.min())
