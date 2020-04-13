class Series:
    def __init__(self, model, style, label):
        self.model = model
        self.style = style
        self.label = label
        self.data = None

    def __str__(self):
        return f'{self.model}\n{self.style} - {self.label}'

    def __repr__(self):
        return self.__str__()
