from apiflask import Schema
from apiflask.fields import Integer, Number, String


class DetectObject(Schema):
    id = Integer()
    x1 = Integer()
    x2 = Integer()
    y1 = Integer()
    y2 = Integer()
    obj_type = String() # enum m.b
    