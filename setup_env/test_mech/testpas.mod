: Minimal passive mechanism used only to check that .mod compilation works.
NEURON {
    SUFFIX testpas
    NONSPECIFIC_CURRENT i
    RANGE g, e
}
UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
}
PARAMETER {
    g = 0.001 (mho/cm2)
    e = -65 (mV)
}
ASSIGNED {
    v (mV)
    i (mA/cm2)
}
BREAKPOINT {
    i = g * (v - e)
}
