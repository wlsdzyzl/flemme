
### if you want to use this script, make sure that you have correctly installed mitsuba, OpenEXR and imath

import numpy as np
from flemme.utils import load_ply, load_img, save_img, save_ply, rotate_by_axis_angle
from flemme.color_table import color_table
import os
from flemme.logger import get_logger
import sys, getopt
logger = get_logger('scripts.render_mesh')
def standardize_bbox(pcl):
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    logger.info("(After standardizing) Center: {}, Scale: {}".format(center, scale))
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
            <integer name="width" value="{}"/>
            <integer name="height" value="{}"/>
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
            <scale x="7" y="6" z="1"/>
            <lookat origin="4, -2, 10" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="7"/>
        </emitter>
    </shape>
</scene>
"""

def main(argv):
    mesh_file = None
    xyz_angles = [0, 0, 0]
    output_path = 'mitsuba_scene.png'
    size=[160, 120]
    opts, args = getopt.getopt(argv, "hi:o:", ['help', 'input_mesh=', 'output_path=', 
                            'xyz_angles=', 'size=', 'mirror_flip=', 'float_height='])
    float_height = 0.1
    mirror_flip = []
    if len(opts) == 0:
        logger.error('unknow options, usage: render_mesh.py -i <input_mesh> -o <output_path=mitsuba_scene.png> --xyz_angles <xyz_angles=0,0,0> --size <size=160,120> --mirror_flip <mirror_flip=None> --float_height <float_height=0.1>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('unknow options, usage: render_mesh.py -i <input_mesh> -o <output_path=mitsuba_scene.png> --xyz_angles <xyz_angles=0,0,0> --size <size=160,120> --mirror_flip <mirror_flip=None> --float_height <float_height=0.1>')
            sys.exit()
        elif opt in ("-i", '--input_mesh'):
            mesh_file = arg
        elif opt in ("-o", '--output_path'):
            output_path = arg
        elif opt in ('--xyz_angles'):
            xyz_angles = [float(a) for a in arg.split(',')]
        elif opt in ('--size'):
            size = [int(s) for s in arg.split(',')]
        elif opt in ('--mirror_flip'):
            mirror_flip = arg.split(',')
        elif opt in ('--float_height'):
            float_height = float(arg)
        else:
            logger.error('unknow options, usage: render_mesh.py -i <input_mesh> -o <output_path=mitsuba_scene.png> --xyz_angles <xyz_angles=0,0,0> --size <size=160,120> --mirror_flip <mirror_flip=None> --float_height <float_height=0.1>')
            sys.exit()
    xml_segments = [xml_head.format(*size)]
    pcl, faces = load_ply(mesh_file, with_faces= True)
    pcl = rotate_by_axis_angle(pcl, xyz_angles[0], xyz_angles[1], xyz_angles[2])
    if len(mirror_flip):
        flip_face = False
        for axis in mirror_flip:
            flip_face = not flip_face
            if axis == 'x':
                pcl[:, 0] = -pcl[:, 0]
            elif axis == 'y':
                pcl[:, 1] = -pcl[:, 1]
            elif axis == 'z':
                pcl[:, 2] = -pcl[:, 2]
            else:
                logger.error('Unknow flip axis, should be one of [x, y, z]')
                exit(1)
        if flip_face:
            faces = faces[:, [2, 1, 0]]
    pcl = standardize_bbox(pcl)
    pcl[:, 2] += float_height

    save_ply('./normalized_mesh.ply', pcl, faces = faces)
    color = [color_table[6][0], color_table[6][1], color_table[6][2]]
    xml_segments.append(xml_ply.format(*color))

    xml_segments.append(xml_tail)

    xml_content = str.join('', xml_segments)

    with open('mitsuba_scene.xml', 'w') as f:
        f.write(xml_content)

    os.system('mitsuba mitsuba_scene.xml')
    img = load_img('mitsuba_scene.exr')
    save_img(output_path, img)
if __name__ == "__main__":
    main(sys.argv[1:])