from .isp import isp_process


def process(bayer_float01):
    return isp_process(bayer_float01)


