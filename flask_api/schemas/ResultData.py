from apiflask import Schema
from apiflask.fields import List, Float

class ResultData(Schema):
    x = List(Float())
    y = List(Float())
    v_x = List(Float())
    v_y = List(Float())
    v = List(Float())
    a_x = List(Float())
    a_y = List(Float())
    a = List(Float())
    time = List(Float())

    F_x = List(Float())
    F_y = List(Float())
    F = List(Float())

    p_x = List(Float())
    p_y = List(Float())
    p = List(Float())

    Ek = List(Float())
    Ep = List(Float())

    err_x = List(Float())
    err_y = List(Float())
    err_v = List(Float())