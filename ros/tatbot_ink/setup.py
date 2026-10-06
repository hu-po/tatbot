from setuptools import find_packages, setup

setup(
    name="tatbot_ink",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/tatbot_ink']),
        ('share/tatbot_ink', ['package.xml']),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hu-po",
    maintainer_email="hello@tatbot.ai",
    description="Acquired DBV3 paths -> tatbot program (rigid placement, exact inks, timed chunks, preview). Pure Python, no ROS import.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": []},
)
