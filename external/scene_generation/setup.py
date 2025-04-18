from setuptools import setup, find_packages

print("PACKAGES FOUND:", find_packages())

# setup(
#     name="scene_generation",
#     version="0.1.0",
#     packages=find_packages(),  # This will include 'scene_generation'
#     include_package_data=True,
#     zip_safe=False,
# )
setup(
    name="scene_generation",
    version="0.1.0",
    package_dir={"": "scene_generation/src"},
    packages=find_packages("scene_generation/src"),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "plyfile==1.1",
        "open3d==0.18.0",
        "Flask==3.0.3",
        "dash==2.14.2",
        "kornia==0.7.1",
        "py360convert==1.0.1",
        "openexr_numpy==0.0.8"
        # Add more as needed
    ],
)
