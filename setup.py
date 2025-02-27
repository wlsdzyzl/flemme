import glob
import os
import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from flemme.config import module_config
exec(open('flemme/__version__.py').read())
install_pcd_modules = module_config['point-cloud']
ext_kwargs = {}
if install_pcd_modules:
    flemme_abspath = osp.dirname(osp.abspath(__file__))
    pcd_ops_root = osp.join('flemme/cpp_extension', "pcd_ops")
    pcd_ops_sources = glob.glob(osp.join(pcd_ops_root, "src", "*.cpp")) + glob.glob(
        osp.join(pcd_ops_root, "src", "*.cu")
    )

    cd_root = osp.join('flemme/cpp_extension', "chamfer_distance")
    cd_sources = [osp.join(cd_root, "chamfer_distance.cpp"), 
        osp.join(cd_root, 'chamfer_distance_cuda.cu')]

    emd_root = osp.join('flemme/cpp_extension', "emd")
    emd_sources = [osp.join(emd_root, "emd.cpp"), 
        osp.join(emd_root, 'emd_cuda.cu')]

    ext_kwargs["ext_modules"]=[
        CUDAExtension(
            name="cpp_extension.pcd_ops",
            sources=pcd_ops_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
            },
            include_dirs=[osp.join(flemme_abspath, pcd_ops_root, "include")],
        ),
        CUDAExtension(
            name="cpp_extension.cd",
            sources=cd_sources,
        ),
        CUDAExtension(
            name="cpp_extension.emd",
            sources=emd_sources,
        ),
    ]
    ext_kwargs["cmdclass"] = {"build_ext": BuildExtension}

os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"

setup(
    name="flemme",
    packages=find_packages(exclude=["tests"]),
    version=__version__,
    author="Guoqing Zhang, Jingyun Yang",
    license="MIT",
    python_requires='>=3.7', 
    include_package_data=True,
    entry_points={'console_scripts': [
        # train
        'train_flemme=flemme.train_flemme:main',    
        # test
        'test_flemme=flemme.test_flemme:main',
        # eval
        'eval_flemme=flemme.eval_flemme:main']
        },
    **ext_kwargs
)
