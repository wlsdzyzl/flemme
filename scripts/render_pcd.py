
### part of this code is adopted from https://github.com/zekunhao1995/PointFlowRenderer/tree/master
import numpy as np
from flemme.utils import load_ply, load_img, save_img, rotate_by_axis_angle
from flemme.color_table import color_table
import os
from flemme.logger import get_logger
import sys, getopt
logger = get_logger('scripts.render_pcd')
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

xml_ball_segment = \
"""
    <shape type="sphere">
        <float name="radius" value="{}"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
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
            <scale x="10" y="12" z="1"/>
            <lookat origin="4, -1, 20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""
def main(argv):
    pcd_file = None
    xyz_angles = []
    output_path = 'mitsuba_scene.png'
    size=[160, 120]
    opts, args = getopt.getopt(argv, "hi:o:", ['help', 'input_pcd=', 'output_path=', 
                            'xyz_angles=', 'size=', 'mirror_flip=', 'float_height=',
                            'point_size=', 'sphere_radius=', 'color_id=',
                            'center=', 'scaling=', 'non_standardized'])
    float_height = 0.1
    point_size = 2048
    sphere_radius = 0.004
    mirror_flip = []
    color_id = 0
    center = np.zeros(3)
    scaling = 1.0
    standardize = True
    if len(opts) == 0:
        logger.error('unknow options, usage: render_pcd.py -i <input_pcd> -o <output_path=mitsuba_scene.png> --xyz_angles <xyz_angles=''> --size <size=160,120> --mirror_flip <mirror_flip=None> --float_height <float_height=0.1> --point_size <point_size=2048> --sphere_radius <sphere_radius=0.004> --color_id <color_id=0> --non_standardized --center <center=0,0,0> --scaling <scaling=1.0>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('unknow options, usage: render_pcd.py -i <input_pcd> -o <output_path=mitsuba_scene.png> --xyz_angles <xyz_angles=''> --size <size=160,120> --mirror_flip <mirror_flip=None> --float_height <float_height=0.1> --point_size <point_size=2048> --sphere_radius <sphere_radius=0.004> --color_id <color_id=0> --non_standardized --center <center=0,0,0> --scaling <scaling=1.0>')
            sys.exit()
        elif opt in ("-i", '--input_pcd'):
            pcd_file = arg
        elif opt in ("-o", '--output_path'):
            output_path = arg
        elif opt in ('--xyz_angles'):
            xyz_angles = [a.split('/') for a in arg.split(',')]
        elif opt in ('--size'):
            size = [int(s) for s in arg.split(',')]
        elif opt in ('--mirror_flip'):
            mirror_flip = arg.split(',')
        elif opt in ('--float_height'):
            float_height = float(arg)
        elif opt in ('--point_size'):
            point_size = int(arg)
        elif opt in ('--sphere_radius'):
            sphere_radius = float(arg)
        elif opt in ('--color_id'):
            color_id = int(arg)
        elif opt in ('--non_standardized'):
            standardize = False
        elif opt in ('--center'):
            center = np.array([float(c) for c in arg.split(',')])
        elif opt in ('--scaling'):
            scaling = float(arg)
        else:
            logger.error('unknow options, usage: render_pcd.py -i <input_pcd> -o <output_path=mitsuba_scene.png> --xyz_angles <xyz_angles=''> --size <size=160,120> --mirror_flip <mirror_flip=None> --float_height <float_height=0.1> --point_size <point_size=2048> --sphere_radius <sphere_radius=0.004> --color_id <color_id=0> --non_standardized --center <center=0,0,0> --scaling <scaling=1.0>')
            sys.exit()
    xml_segments = [xml_head.format(*size)]

    pcl = load_ply(pcd_file)
    
    if point_size < pcl.shape[0]:
        pcl = pcl[np.random.choice(range(pcl.shape[0]), point_size, replace=False)]

    pcl_color = None
    if pcl.shape[1] == 6:
        pcl_color = pcl[..., 3:6] / 255.0
        pcl = pcl[..., :3]
    if  sum([len(_) == 1 for _ in xyz_angles]) == len(xyz_angles) and len(xyz_angles) % 3 == 0:
        xyz_angles = [float(a[0]) for a in xyz_angles]
        for i in range(0, len(xyz_angles), 3):
            pcl = rotate_by_axis_angle(pcl, x_angle=xyz_angles[i],
                y_angle = xyz_angles[i+1],
                z_angle = xyz_angles[i+2])
    else:
        for aa in xyz_angles:
            axis, angle = aa
            angle = float(angle)
            if axis == 'x':
                pcl = rotate_by_axis_angle(pcl, x_angle = angle)
            elif axis == 'y':
                pcl = rotate_by_axis_angle(pcl, y_angle = angle)
            elif axis == 'z':
                pcl = rotate_by_axis_angle(pcl, z_angle = angle)
    if len(mirror_flip):
        for axis in mirror_flip:
            if axis == 'x':
                pcl[:, 0] = -pcl[:, 0]
            elif axis == 'y':
                pcl[:, 1] = -pcl[:, 1]
            elif axis == 'z':
                pcl[:, 2] = -pcl[:, 2]
            else:
                logger.error('Unknow flip axis, should be one of [x, y, z]')
                exit(1)
    if standardize:
        pcl = standardize_bbox(pcl)
    else:
        pcl = ((pcl - center)/scaling).astype(np.float32)
    pcl[:, 2] += float_height
    

    if pcl_color is not None:
        for i in range(pcl.shape[0]):
            xml_segments.append(xml_ball_segment.format(sphere_radius, pcl[i,0],pcl[i,1],pcl[i,2], 
                pcl_color[i,0],pcl_color[i,1],pcl_color[i,2]))
    else:
        for i in range(pcl.shape[0]):
            color = color_table[color_id].tolist()
            xml_segments.append(xml_ball_segment.format(sphere_radius, pcl[i,0],pcl[i,1],pcl[i,2], *color))

    xml_segments.append(xml_tail)
    xml_content = str.join('', xml_segments)

    with open('mitsuba_scene.xml', 'w') as f:
        f.write(xml_content)

    os.system('mitsuba mitsuba_scene.xml')
    img = load_img('mitsuba_scene.exr')
    save_img(output_path, img)
if __name__ == "__main__":
    main(sys.argv[1:])