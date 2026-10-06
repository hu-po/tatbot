from setuptools import find_packages, setup

setup(
    name="tatbot_motion",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/tatbot_motion']),
        ('share/tatbot_motion', ['package.xml']),
        ('share/tatbot_motion/config', ['config/motion.yaml']),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hu-po",
    maintainer_email="hello@tatbot.ai",
    description="Program op -> timed Cartesian samples -> joint trajectory (CLIK on pinocchio). Pure Python, no ROS import.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": []},
)
