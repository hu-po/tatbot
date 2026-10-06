from setuptools import find_packages, setup

setup(
    name="tatbot_calib",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/tatbot_calib']),
        ('share/tatbot_calib', ['package.xml']),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hu-po",
    maintainer_email="hello@tatbot.ai",
    description="Probe-station calibration: the station measured fresh from its fiducials.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": ["station = tatbot_calib.cli:main", "calib = tatbot_calib.program:main",
                                        "register = tatbot_calib.register:main", "sweep = tatbot_calib.sweep:main",
                                        "chain = tatbot_calib.chain:main"]},
)
