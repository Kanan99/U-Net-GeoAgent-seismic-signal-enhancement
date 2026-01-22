class SignalEnhancer:
    def __init__(self, model):
        self.model = model

    def enhance(self, patch_tensor):
        return self.model(patch_tensor)
