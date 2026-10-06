from setuptools import find_packages, setup

setup(
    name="tatbot_rerun",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/tatbot_rerun']),
        ('share/tatbot_rerun', ['package.xml']),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hu-po",
    maintainer_email="hello@tatbot.ai",
    description="ROS -> Rerun bridge for the fleet viewer (through scripts/lib/tatbot_rerun.py, the only rr.init).",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": [
            'rerun_bridge = tatbot_rerun.node:main',
        ]},
)
