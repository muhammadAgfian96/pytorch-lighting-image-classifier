import timm
import uuid

def is_uuid(name):
    # Check if the name is a timm model
    name_model = name.replace("timm/", "")
    if name_model in timm.list_models():
        return True
    
    # Check if the name is a valid UUID
    try:
        uuid.UUID(name, version=4)
        return True
    except ValueError:
        return False

# Example usage:
print(is_uuid("resnet50"))  # True (timm model)
print(is_uuid("my_model"))  # False (not a timm model or UUID)
print(is_uuid("65d31f59434b4da8bb1b1b22b5339c97"))  # False (not a timm model or UUID)
print(is_uuid(uuid.uuid4().hex))  # True (UUID)
