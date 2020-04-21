from distutils.version import StrictVersion
import requests


exec(open('auditai/version.py').read())
PACKAGE_NAME = 'audit-AI'
VERSION = StrictVersion(__version__)  # noqa


def versions(package_name):
    url = "https://pypi.python.org/pypi/{}/json".format(package_name)
    response = requests.get(url)

    data = response.json()
    versions = [StrictVersion(v) for v in data["releases"].keys()]
    versions.sort()
    return versions


print('Version: ', VERSION)
if not VERSION > versions(PACKAGE_NAME)[-1]:
    print("Version would not be the newest version.")
    exit(1)

print("Version would be the newest version.")
