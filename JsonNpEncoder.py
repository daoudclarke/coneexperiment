# Bismillahi-r-Rahmani-r-Rahim

# Allow numpy arrays to be encoded in json

from json import JSONEncoder

class JsonNpEncoder(JSONEncoder):
    def default(self, o):
        try:
            l = o.tolist()
        except AttributeError:
            pass
        else:
            return l
        return JSONEncoder.default(self, o)
