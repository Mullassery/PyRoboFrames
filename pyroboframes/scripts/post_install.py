"""Post-install messaging for PyRoboFrames"""


def post_install():
    print(
        """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ PyRoboFrames installed successfully!

📌 WHAT IS THIS?
   Multi-rate sensor fusion & I/O

🚀 GET STARTED:
   $ python3 -c "from pyroboframes import *; print('PyRoboFrames ready')"
   $ python3 -c "import pyroboframes; print(f'v{pyroboframes.__version__ if hasattr(pyroboframes, \"__version__\") else \"latest\"}')"

📖 DOCUMENTATION:
   Repo:     https://github.com/Mullassery/PyRoboFrames
   Tutorials: https://github.com/Mullassery/PyRoboFrames#readme
   Issues:    https://github.com/Mullassery/PyRoboFrames/issues

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    )


if __name__ == "__main__":
    post_install()
