from apiflask import Schema
from apiflask.fields import String, Nested
from schemas.ResultData import ResultData

class Result(Schema):
    status = String(required=True)
    data = Nested(ResultData)
    error_message = String()