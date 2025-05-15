from mongoengine import Document, EmbeddedDocument, fields, CASCADE
import datetime


class TemplateSession(Document):
    user_id = fields.StringField(required=True)
    created_at = fields.DateTimeField(default = datetime.datetime.utcnow)
    template_type = fields.StringField(required=True)
    image_path = fields.StringField(required=True)
    is_verified = fields.BooleanField(default = False)

    def __str__(self):
        return f"Session for {self.user_id}"

class Symbol(Document):
    session = fields.ReferenceField(TemplateSession, reverse_delete_rule=CASCADE)
    label = fields.StringField(required=True)
    predicted = fields.StringField(null=True)
    image_path = fields.StringField(required=True)
    x = fields.IntField()
    y = fields.IntField()
    width = fields.IntField()
    height = fields.IntField()
    is_corrected = fields.BooleanField(default=False)

    def __str__(self):
        return f"'{self.label}' at ({self.x}, {self.y})"


class WordSample(Document):
    session = fields.ReferenceField(TemplateSession, reverse_delete_rule=CASCADE)
    text = fields.StringField(required=True)
    predicted = fields.StringField(null=True)
    image_path = fields.StringField(required=True)
    x = fields.IntField()
    y = fields.IntField()
    width = fields.IntField()
    height = fields.IntField()
    is_corrected = fields.BooleanField(default=False)