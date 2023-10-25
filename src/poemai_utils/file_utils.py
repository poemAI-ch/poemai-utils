import os

import pkg_resources


def _calc_package_and_path(sub_package_name, filename):
    root_package_name = sub_package_name.split(".")[0]
    relative_path = os.path.join(*sub_package_name.split(".")[1:], filename)
    return root_package_name, relative_path


def get_resource_path(sub_package_name, filename):
    root_package_name, relative_path = _calc_package_and_path(
        sub_package_name, filename
    )
    return pkg_resources.resource_filename(root_package_name, relative_path)


def get_resource_string(sub_package_name, filename):
    root_package_name, relative_path = _calc_package_and_path(
        sub_package_name, filename
    )
    return pkg_resources.resource_string(root_package_name, relative_path).decode(
        "utf-8"
    )
