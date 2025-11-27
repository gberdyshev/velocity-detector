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