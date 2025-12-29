
### part of this code is adopted from https://github.com/zekunhao1995/PointFlowRenderer/tree/master
import numpy as np
from flemme.utils import load_ply, load_img, save_img
from flemme.color_table import color_table
import os
from flemme.logger import get_logger
logger = get_logger('scripts.render_pcd')
def standardize_bbox(pcl, points_per_object, pcl_color = None):
    if points_per_object > 0:
        pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    else:
        pt_indices = np.arange(pcl.shape[0])
    pcl = pcl[pt_indices] # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    logger.info("(After standardizing) Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
    if pcl_color is not None:
        pcl_color = pcl_color[pt_indices]
    return result, pcl_color

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
            <lookat origin="2.5,-5,2" target="0,0,0" up="0,0,1"/>
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
        <float name="radius" value="0.004"/>
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

def colormap(x,y,z):
    # vec = np.array([x,y,z])
    # vec = np.clip(vec, 0.001,1.0)
    # norm = np.sqrt(np.sum(vec**2))
    # vec /= norm
    # return [vec[0], vec[1], vec[2]]
    return [color_table[6][0], color_table[6][1], color_table[6][2]]
    # return [color_table[1][0], color_table[1][1], color_table[1][2]]
def main(argv)
    pcd_file = None
    xyz_angles = [0, 0, 0]
    output_path = 'mitsuba_scene.png'
    size=[160, 120]
    opts, args = getopt.getopt(argv, "hi:o:", ['help', 'input_pcd=', 'output_path=', 
                            'xyz_angles=', 'size='])

    if len(opts) == 0:
        logger.error('unknow options, usage: render_pcd.py -i <input_pcd> -o <output_path=mitsuba_scene.png> --xyz_angles <xyz_angles=0,0,0> --size <size=160,120>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('unknow options, usage: render_pcd.py -i <input_pcd> -o <output_path=mitsuba_scene.png> --xyz_angles <xyz_angles=0,0,0> --size <size=160,120>')
            sys.exit()
        elif opt in ("-i", '--input_pcd'):
            pcd_file = arg
        elif opt in ("-o", '--output_path'):
            output_path = arg
        elif opt in ('--xyz_angles'):
            xyz_angles = [int(a) for a in arg.split(',')]
        elif opt in ('--size'):
            size = [int(s) for s in arg.split(',')]
        else:
            logger.error('unknow options, usage: render_pcd.py -i <input_pcd> -o <output_path=mitsuba_scene.png> --xyz_angles <xyz_angles=0,0,0> --size <size=160,120>')
            sys.exit()
    xml_segments = [xml_head.format(*size)]

    pcl = load_ply(pcd_file)
    pcl_color = None
    if pcl.shape[1] == 6:
        pcl_color = pcl[..., 3:6] / 255.0
        pcl = pcl[..., :3]
    pcl = rotate_by_axis_angle(pcl, xyz_angles[0], xyz_angles[1], xyz_angles[2])
    pcl, pcl_color = standardize_bbox(pcl, 512, pcl_color)

    pcl[:,2] +=0.05
    

    if pcl_color is not None:
        for i in range(pcl.shape[0]):
            xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], 
                pcl_color[i,0],pcl_color[i,1],pcl_color[i,2]))
    else:
        for i in range(pcl.shape[0]):
            color = colormap(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5-0.0125)
            xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))

    xml_segments.append(xml_tail)
    xml_content = str.join('', xml_segments)

    with open('mitsuba_scene.xml', 'w') as f:
        f.write(xml_content)

    os.system('mitsuba mitsuba_scene.xml')
    img = load_img('mitsuba_scene.exr')
    save_img(output_path, img)