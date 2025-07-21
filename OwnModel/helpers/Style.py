class Style:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

    @staticmethod
    def make_color(color, *text):
        text = list(text)
        text = map(lambda t: str(t), text)
        return color + " ".join(text) + Style.RESET

    @staticmethod
    def black(*text):
        return Style.make_color(Style.BLACK, *text)

    @staticmethod
    def red(*text):
        return Style.make_color(Style.RED, *text)

    @staticmethod
    def green(*text):
        return Style.make_color(Style.GREEN, *text)

    @staticmethod
    def yellow(*text):
        return Style.make_color(Style.YELLOW, *text)

    @staticmethod
    def blue(*text):
        return Style.make_color(Style.BLUE, *text)

    @staticmethod
    def magenta(*text):
        return Style.make_color(Style.MAGENTA, *text)

    @staticmethod
    def cyan(*text):
        return Style.make_color(Style.CYAN, *text)

    @staticmethod
    def white(*text):
        return Style.make_color(Style.WHITE, *text)

    @staticmethod
    def underline(*text):
        return Style.UNDERLINE + text.join(" ") + Style.RESET

    @staticmethod
    def overview(text = "Hello World!"):
        print(Style.black(text))
        print(Style.red(text))
        print(Style.green(text))
        print(Style.yellow(text))
        print(Style.magenta(text))
        print(Style.cyan(text))
        print(Style.white(text))
        print(Style.underline(text))