
import numpy as np
from flemme.utils import load_ply, load_img, save_img, save_ply
from flemme.color_table import color_table
import os

def standardize_bbox(pcl):
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
    return result

xml_head = \
"""
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="0,-5,2" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        
        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1600"/>
            <integer name="height" value="1200"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
            <string name="file_format" value="openexr"/>
            <string name="component_format" value="float32"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
    
"""

xml_ply = \
"""
    <shape type="ply">
        <string name="filename" value="./normalized_mesh.ply"/>
        <transform name="to_world">
            <scale value="1"/>
            <translate x="0" y="0" z="0"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>

"""

xml_tail = \
"""
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="6" y="6" z="1"/>
            <lookat origin="4, -1.5, 10" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="7"/>
        </emitter>
    </shape>
</scene>
"""


xml_segments = [xml_head]

mesh_file = "/media/wlsdzyzl/DATA1/datasets/pcd/imageCAS/output/mesh_from_xyzr_ours_fast/10018985.ply"
pcl, faces = load_ply(mesh_file, with_faces= True)

pcl = pcl[:,[2, 1, 0]]
# pcl[:, 1] = -pcl[:, 1]
faces = faces[:, [2, 1, 0]]
pcl = standardize_bbox(pcl)
pcl[:, 2] += 0.0

save_ply('./normalized_mesh.ply', pcl, faces = faces)
color = [color_table[6][0], color_table[6][1], color_table[6][2]]
xml_segments.append(xml_ply.format(*color))

xml_segments.append(xml_tail)

xml_content = str.join('', xml_segments)

with open('mitsuba_scene.xml', 'w') as f:
    f.write(xml_content)

os.system('mitsuba mitsuba_scene.xml')
img = load_img('mitsuba_scene.exr')
save_img('mitsuba_scene.png', img)