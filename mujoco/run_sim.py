import mujoco
import mujoco.viewer

# Load the model
model = mujoco.MjModel.from_xml_path("tentacle.xml")
data = mujoco.MjData(model)

# Launch the viewer (pauses code execution until you close the window)
mujoco.viewer.launch(model, data)
