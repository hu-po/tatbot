from setuptools import find_packages, setup

setup(
    name="tatbot_bridge",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/tatbot_bridge']),
        ('share/tatbot_bridge', ['package.xml']),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hu-po",
    maintainer_email="hello@tatbot.ai",
    description="zenoh and ROS: stencild page poses on the tatbot bus become /tatbot/page and tf page/PATTERN_ID; /joint_states rides the bus as tatbot.arm-joints/1.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": [
            'bridge = tatbot_bridge.node:main',
            'page_watch = tatbot_bridge.watch:main',
        ]},
)
