from IPython.display import HTML, display
class Tag:
    def __init__(self, name, value):
        self.name = name
        self.value = value
    def __repr__(self):
        return "<%s>%s</%s>"%(self.name, str(self.value), self.name)
        
class Linear:
    def __init__(self, data):
        self.data = list(data)
    def __repr__(self):
        return ''.join(list(map(str, self.data)))

class Table:
    def __init__(self, cols):
        self.header = list(iter(cols))
        self.len = len(self.header)
        self.contents = []
        self.compiled = False
    def add(self, data):
        if(len(data) != self.len):
            raise AssertionError
        self.contents.append(data)
        self.compiled = False
    
    def __repr__(self):
        if self.compiled:
            return self.compiledHTML
        header = Tag("tr", Linear(list(map(lambda x:Tag('th', x), self.header))))
        contents = []
        for c in self.contents:
            contents += [ Tag("tr", Linear(list(map(lambda x:Tag('td', x), c))))]
        
        self.compiledHTML = str(Tag('Table', Linear((header, Linear(contents)))))
        self.compiled = True
        return self.compiledHTML
    
    def display_notebook(self):
        display(HTML(str(self)))