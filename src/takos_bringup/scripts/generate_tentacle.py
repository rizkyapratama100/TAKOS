import math

def generate_sdf():
    # Constants
    NUM_SEGMENTS = 10
    TIP_SCALE = 0.25
    SCALE_RATIO = 1.1  # Each segment is 1.1x larger than the one above it
    UNIT_HEIGHT = 5.0  # Height for scale=1.0
    UNIT_MASS = 1.0    # Mass for scale=1.0
    
    # Calculate scales from Tip (smallest) to Base (largest)
    scales = []
    current_scale = TIP_SCALE
    for _ in range(NUM_SEGMENTS):
        scales.append(current_scale)
        current_scale *= SCALE_RATIO
    
    scales.reverse() # Base to Tip
    
    # Calculate positions
    positions = []
    heights = []
    
    current_z = 1.5
    for i in range(NUM_SEGMENTS):
        scale = scales[i]
        height = scale * UNIT_HEIGHT
        heights.append(height)
        
        if i == 0:
            positions.append(1.5)
        else:
            prev_z = positions[i - 1]
            prev_h = heights[i - 1]
            curr_h = height
            d = ((prev_h + curr_h) / 2.0) * 0.95  # slight overlap
            positions.append(prev_z + d)
            
    # Generate XML
    xml = []
    xml.append('<?xml version="1.0" ?>')
    xml.append('<sdf version="1.7">')
    xml.append('  <world name="default">')
    xml.append('    <include>')
    xml.append('      <uri>model://ground_plane</uri>')
    xml.append('    </include>')
    xml.append('')
    xml.append('    <model name="tentacle_robot">')
    
    scale_names = ["seg_{}".format(i+1) for i in range(NUM_SEGMENTS)]
    
    for i in range(NUM_SEGMENTS):
        name = scale_names[i]
        scale = scales[i]
        pos_z = positions[i]
        
        mass = UNIT_MASS * (scale ** 3)
        inertia = 1.0 * (scale ** 3) 
        
        xml.append(f'      <link name="{name}_link">')
        xml.append(f'        <pose>0 0 {pos_z:.4f} 0 0 0</pose>')
        xml.append('        <inertial>')
        xml.append(f'          <mass>{mass:.4f}</mass>')
        xml.append('          <inertia>')
        xml.append(f'            <ixx>{inertia:.4f}</ixx> <ixy>0</ixy> <ixz>0</ixz>')
        xml.append(f'            <iyy>{inertia:.4f}</iyy> <iyz>0</iyz>')
        xml.append(f'            <izz>{inertia:.4f}</izz>')
        xml.append('          </inertia>')
        xml.append('        </inertial>')
        
        color = 'Gazebo/Orange' if i % 2 == 0 else 'Gazebo/DarkGrey'
        xml.append(f'        <visual name="{name}_visual">')
        xml.append('          <pose>0 0 0 0 1.57 0</pose>')
        xml.append('          <geometry>')
        xml.append('            <mesh>')
        xml.append('              <uri>model://segment/meshes/segment.stl</uri>')
        xml.append(f'              <scale>{scale:.4f} {scale:.4f} {scale:.4f}</scale>')
        xml.append('            </mesh>')
        xml.append('          </geometry>')
        xml.append('          <material>')
        xml.append('            <script>')
        xml.append('              <uri>file://media/materials/scripts/gazebo.material</uri>')
        xml.append(f'              <name>{color}</name>')
        xml.append('            </script>')
        xml.append('          </material>')
        xml.append('        </visual>')
        
        # Collision scaled down slightly (0.9) to prevent coplanar interpenetration at joints
        c_scale = scale * 0.9 
        xml.append(f'        <collision name="{name}_collision">')
        xml.append('          <pose>0 0 0 0 1.57 0</pose>')
        xml.append('          <geometry>')
        xml.append('            <mesh>')
        xml.append('              <uri>model://segment/meshes/segment.stl</uri>')
        xml.append(f'              <scale>{c_scale:.4f} {c_scale:.4f} {c_scale:.4f}</scale>')
        xml.append('            </mesh>')
        xml.append('          </geometry>')
        xml.append('        </collision>')
        xml.append('      </link>')
        xml.append('')
        
    xml.append('      <joint name="world_fixed" type="fixed">')
    xml.append('        <parent>world</parent>')
    xml.append(f'        <child>{scale_names[0]}_link</child>')
    xml.append('      </joint>')
    xml.append('')

    for i in range(NUM_SEGMENTS - 1):
        parent = scale_names[i]
        child = scale_names[i+1]
        inter = f"{child}_inter"
        
        # Current logic:
        # P_parent + H_parent/2 = Top of Parent
        # Lower pivot by 25% of height to hide gaps
        
        parent_top_z = positions[i] + heights[i] / 2.0
        pivot_offset = heights[i] * 0.25
        pivot_z = parent_top_z - pivot_offset
        
        # Intermediate link at the pivot point
        xml.append(f'      <link name="{inter}">')
        xml.append(f'        <pose>0 0 {pivot_z:.4f} 0 0 0</pose>')
        xml.append('        <inertial>')
        xml.append('          <mass>0.001</mass>')
        xml.append('          <inertia>')
        xml.append('            <ixx>0.0001</ixx> <ixy>0</ixy> <ixz>0</ixz>')
        xml.append('            <iyy>0.0001</iyy> <iyz>0</iyz>')
        xml.append('            <izz>0.0001</izz>')
        xml.append('          </inertia>')
        xml.append('        </inertial>')
        xml.append('      </link>')
        xml.append('')

        # 1. Bend Joint (Revolute): Parent -> Intermediate
        xml.append(f'      <joint name="joint_{i}_{i+1}_bend" type="revolute">')
        xml.append(f'        <parent>{parent}_link</parent>')
        xml.append(f'        <child>{inter}</child>')
        xml.append('        <pose>0 0 0 0 0 0</pose>')
        xml.append('        <axis>')
        xml.append('          <xyz>1 0 0</xyz>') # X-axis rotation
        xml.append('          <limit>')
        xml.append('            <lower>-0.52</lower>')
        xml.append('            <upper>0.52</upper>')
        xml.append('          </limit>')
        xml.append('          <dynamics>')
        xml.append('             <spring_stiffness>50.0</spring_stiffness>')
        xml.append('             <spring_reference>0</spring_reference>')
        xml.append('             <damping>5.0</damping>')
        xml.append('          </dynamics>')
        xml.append('        </axis>')
        xml.append('      </joint>')
        xml.append('')

        # 2. Compress Joint (Prismatic): Intermediate -> Child
        # Child center is positions[i+1]
        # Pivot in Child frame (Offset from center): (Pivot_Z - Child_center)
        compress_pose_z = pivot_z - positions[i+1]
        
        xml.append(f'      <joint name="joint_{i}_{i+1}_compress" type="prismatic">')
        xml.append(f'        <parent>{inter}</parent>')
        xml.append(f'        <child>{child}_link</child>')
        xml.append(f'        <pose>0 0 {compress_pose_z:.4f} 0 0 0</pose>')
        xml.append('        <axis>')
        xml.append('          <xyz>0 0 1</xyz>') # Z-axis (local)
        xml.append('          <limit>')
        xml.append('            <lower>-1.0</lower>') # Can squish
        xml.append('            <upper>0.1</upper>') # Slightly loose
        xml.append('          </limit>')
        xml.append('          <dynamics>')
        xml.append('             <spring_stiffness>500.0</spring_stiffness>')
        xml.append('             <spring_reference>0</spring_reference>')
        xml.append('             <damping>50.0</damping>')
        xml.append('          </dynamics>')
        xml.append('        </axis>')
        xml.append('      </joint>')
        xml.append('')
        
        # Add Controllers for both
        for suffix in ["bend", "compress"]:
            xml.append('      <plugin filename="gz-sim-joint-position-controller-system"')
            xml.append('              name="gz::sim::systems::JointPositionController">')
            xml.append(f'        <joint_name>joint_{i}_{i+1}_{suffix}</joint_name>')
            xml.append('        <p_gain>100</p_gain>')
            xml.append('        <i_gain>0.1</i_gain>')
            xml.append('        <d_gain>1.0</d_gain>')
            xml.append('        <cmd_max>1000</cmd_max>')
            xml.append('        <cmd_min>-1000</cmd_min>')
            xml.append('      </plugin>')
            xml.append('')

    # Add JointStatePublisher
    xml.append('      <plugin filename="gz-sim-joint-state-publisher-system"')
    xml.append('              name="gz::sim::systems::JointStatePublisher">')
    xml.append('      </plugin>')
        
    xml.append('    </model>')
    xml.append('  </world>')
    xml.append('</sdf>')
    
    return "\n".join(xml)

if __name__ == "__main__":
    content = generate_sdf()
    
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resource_dir = os.path.join(script_dir, '..', 'resource')
    output_path = os.path.join(resource_dir, 'tentacle.sdf')
    
    with open(output_path, "w") as f:
        f.write(content)
    print(f"Generated {output_path}")
