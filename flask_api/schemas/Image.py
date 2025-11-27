from apiflask import Schema
from apiflask.fields import File
from apiflask.validators import FileType, FileSize

#
class Image(Schema):
    image = File(required=True, validate=[FileType(['.png', '.jpg', '.jpeg']), FileSize(max='15 MB')])