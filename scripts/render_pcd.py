
import numpy as np
from flemme.utils import load_ply, load_img, save_img
from flemme.color_table import color_table
import os
from flemme.logger import get_logger
logger = get_logger('scripts::render_pcd')
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

xml_segments = [xml_head]

# pcl = load_ply("/media/wlsdzyzl/DATA1/datasets/pcd/MedPointS/segmentation/fold2/colorized_pcd/024165.ply", vertex_features = ['red', 'green', 'blue'])
# pcl = load_ply("/media/wlsdzyzl/DATA1/flemme-results/seg/MedPointS/036881_colorized_tar.ply", vertex_features = ['red', 'green', 'blue'])
# heart
# pcl = load_ply('/media/wlsdzyzl/DATA1/datasets/pcd/MedPointS/classification/fold1/heart/s0455.ply')
# heart partial
# pcl = load_ply('/media/wlsdzyzl/DATA1/datasets/pcd/MedPointS/completion/fold1/heart/partial/s0455.ply')
# liver
# pcl = load_ply('/media/wlsdzyzl/DATA1/datasets/pcd/MedPointS/classification/fold1/liver/019541.ply')
# liver partial
# pcl = load_ply('/media/wlsdzyzl/DATA1/datasets/pcd/MedPointS/completion/fold1/liver/partial/019541.ply')
# stomach
# pcl = load_ply('/media/wlsdzyzl/DATA1/datasets/pcd/MedPointS/classification/fold1/stomach/s0075.ply')
# stomach partial
# pcl = load_ply('/media/wlsdzyzl/DATA1/datasets/pcd/MedPointS/completion/fold1/stomach/partial/s0075.ply')
# pcl = load_ply("/media/wlsdzyzl/DATA1/flemme-results/cpl/medshapenet/s0300_colon_ours.ply")
# kidney
# pcl = load_ply('/media/wlsdzyzl/DATA1/datasets/pcd/MedPointS/classification/fold1/kidney/s0212.ply')
# pcl = load_ply('/media/wlsdzyzl/DATA1/datasets/pcd/MedPointS/completion/fold1/kidney/partial/s0212.ply')
# spleen


# brain
# pcl = load_ply('/media/wlsdzyzl/DATA1/datasets/pcd/MedPointS/classification/fold1/brain/017214.ply')

# brain
# pcl = load_ply('/media/wlsdzyzl/DATA1/datasets/pcd/MedPointS/classification/fold1/iliacartery/s0157.ply')
# imagecas
pcl = load_ply("/media/wlsdzyzl/DATA1/datasets/pcd/imageCAS/output_lr/surface/10016975_1.ply")
pcl_color = None
if pcl.shape[1] == 6:
    pcl_color = pcl[..., 3:6] / 255.0
    pcl = pcl[..., :3]
pcl, pcl_color = standardize_bbox(pcl, 512, pcl_color)

# heart
# pcl = pcl[:,[0, 2, 1]]

# # liver
# pcl[:,1] *= -1
# pcl[:,2] *= -1

## stomach
# pcl = pcl[:,[0, 2, 1]]
# pcl[:,0] *= -1
# pcl[:,1] *= -1
# pcl[:,2] *= -1

## segmentation
# pcl[:,0] *= -1
# pcl[:,1] *= -1
# pcl[:,2] *= -1

## whole body
# pcl[:,1] *= -1
# pcl[:,2] *= -1
# pcl[:,2] +=0.15

## imagecas
pcl = pcl[:,[1, 0, 2]]
# pcl[:,1] *= -1
pcl[:,2] *= -1
# pcl[:,2] +=0.15

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
save_img('mitsuba_scene.png', img)