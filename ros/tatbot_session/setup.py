from setuptools import find_packages, setup

setup(
    name="tatbot_session",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/tatbot_session']),
        ('share/tatbot_session', ['package.xml']),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hu-po",
    maintainer_email="hello@tatbot.ai",
    description="The orchestrator: Draw/Touch/Land actions, Decide, ledger, page touches, e-stop continue/land, run logs.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": [
            'session = tatbot_session.node:main',
            'client = tatbot_session.client:main',
            'gauge = tatbot_session.gauge:main',
            'keypad = tatbot_session.keypad:main',
        ]},
)
