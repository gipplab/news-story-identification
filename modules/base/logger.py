class Logger():
    def __init__(self):
        self.label = ''
        self.current = 1
        self.goal = 100
        self.newline = False

    def log(self):
        print(f'[{self.label.upper()}] {"{0:0.2f}".format((self.current) / (self.goal) * 100)}% {self.current}/{self.goal}', end='')
        if self.newline:
            print()