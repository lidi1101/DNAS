AUTODEEPLAB = [
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_5x5',
    'dil_conv_3x3',
    #'avg_pool_3x3',
    #'max_pool_3x3',
    'skip_connect',
    'none',
]

DIL3X3 = [
    'sep_conv_3x3',
    'dil_3x3_2',
    'dil_3x3_4',
    'dil_3x3_8',
    'skip_connect',
    'none',
]

DEF = [
        'sep_conv_3x3',
        'dil_conv_3x3',
        'dil_3x3_2',
        'def_3x3',
        'skip_connect',
        'none',
]

CSDD = [
        'con_conv_3x3',
        'sep_conv_3x3',
        'dil_conv_3x3',
        'def_3x3',
        'skip_connect',
        'none',
]


PRIMITIVES = {
    "AutoDeepLab": AUTODEEPLAB,
    "Dil3x3": DIL3X3,
    "Deformable": DEF,
    "CSDD": CSDD,
}
