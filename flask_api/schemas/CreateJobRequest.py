from apiflask import Schema
from apiflask.fields import File, String
from apiflask.validators import FileType, FileSize

class CreateJobRequest(Schema):
    video_file = File(required=True)
    data = String(
        required=True
    )