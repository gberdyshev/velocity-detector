from apiflask import Schema
from apiflask.fields import File, Float, Nested
from apiflask.validators import FileType, FileSize
from schemas.DetectObject import DetectObject

class JobMetadata(Schema):
    selected_frame_time = Float(required=True)
    detect_object = Nested(DetectObject, required=True)
    pixel_size = Float(required=True)