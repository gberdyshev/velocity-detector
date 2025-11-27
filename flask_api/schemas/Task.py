from apiflask import Schema
from apiflask.fields import String, Integer


class Task(Schema):
    id = String()
    created_at = Integer()
