from CSparse3 import __config__
from CSparse3 import csc_numba


def compile_all():
    __config__.NATIVE = False
    csc_numba.compile_code()


if __name__ == '__main__':
    compile_all()
