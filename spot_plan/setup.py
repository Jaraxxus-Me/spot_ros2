from setuptools import setup

package_name = "spot_plan"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="BD AI Institute",
    maintainer_email="engineering@theaiinstitute.com",
    description="Examples of using ROS 2 to control Spot",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "walk_forward = spot_plan.walk_forward:main",
            "arm_simple = spot_plan.arm_simple:main",
            "send_inverse_kinematics_requests = spot_plan.send_inverse_kinematics_requests:main",
            "batch_trajectory = spot_plan.batch_trajectory:main",
            "walk_around = spot_plan.walk_around_localize:main",
            "localize = spot_plan.localize:main",
        ],
    },
)
