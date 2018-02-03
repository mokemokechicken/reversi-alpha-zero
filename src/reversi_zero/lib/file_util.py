import os


def read_as_int(filename):
    if os.path.exists(filename):
        try:
            with open(filename, "rt") as f:
                ret = int(str(f.read()).strip())
                if ret:
                    return ret
        except ValueError:
            pass
