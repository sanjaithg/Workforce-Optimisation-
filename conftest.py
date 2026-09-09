"""Keep unrelated system-wide pytest plugins (e.g. ROS) out of this project's runs."""
import os

os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
