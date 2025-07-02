def write_csv_table(data):
    if len(data) == 0:
        return "empty table"
    return '\n'.join([";".join(d) for d in data])


def write_adjusted_table(data):
    if len(data) == 0:
        return "empty table"
    sizes = [0] * len(data[0])
    for i in range(len(data)):
        for j in range(len(sizes)):
            if not isinstance(data[i][j], str):
                data[i][j] = str(data[i][j])

    for i in range(len(sizes)):
        sizes[i] = max([len(data[row][i]) for row in range(len(data))])
    for i in range(len(data)):
        for j in range(len(sizes)):
            data[i][j] = data[i][j].rjust(sizes[j], " ")
    return '\n'.join([" | ".join(d) for d in data])

def write_latex_table(data, indexing=True):
    if len(data) == 0:
        return "empty table"

    columns_str = "".join(["|c"] * len(data[0])) + "|"
    if indexing:
        columns_str = "|l" + "".join(["|c"] * (len(data[0])-1)) + "|"
    str = ""
    str += "%=============================================================\n"
    str += "%Generated from python\n"
    str += "\\begin{center}\\begin{tabular}{" +columns_str+ "}\n\\hline\n"
    for row in data:
        for i, item in enumerate(row):
            str+=item
            if i != len(row)-1:
                str+="&"
        str+="\\\\\n"
        str+="\\hline\n"
    str += "\\end{tabular}\\end{center}\n"
    str += "%=============================================================\n"
    return str

def transposition(data):
    """
    Transpose table. Change rows with columns
    """
    if len(data) == 0:
        raise ValueError("transposition: Cannot Transpose empty table")
    rows = len(data)
    cols = len(data[0])

    transposed = [["" for r in range(rows)] for c in range(cols)]
    transposed[1][1]="asdad"
    for r in range(rows):
        for c in range(cols):
            print(transposed)
            transposed[c][r] = data[r][c]

    return transposed

class TablePrinter:
    FORMAT_TXT = "txt"
    FORMAT_LATEX = "latex"
    FORMAT_CSV = "csv"
    def __init__(self, *argv, precision=[2,1], format="txt", indexing = True, transposed=False, title=""):
        """
            TablePrinter constructor. It creates an object of table. This object collects headers, indexes and other
            configuration regarding table. While the data are generated the results are stored inside object.
            Object is fit with new data by TablePrinter.append(row, index=None) function. After all data are generated
            object can be printed using selected configuration (format = [txt, latex], precision)
            :param argv: columns to be printed
            :param precision: either list or number. If table cell contains number then it will be presented in selected
                              precision. If list then all specified element shoudl be set as in list. Ifthis is
                              shorter then columns then left columns will be as the last element in precision. If precision
                              is a number then all columns will be formated in the same way.
            :param format: print format adjusted txt for console and latex for documents
            :param indexing: If set to True then numbering column will be printed (and auto generated). If index param set
                             for append method then it can take form of a string. Not specified elements will be numbers
            :param transposed: If set to True table will be transposed while printing.

        """
        self._columns = list(argv)
        self._indexes = []
        self._title = title
        self._indexing = indexing
        self._transposed = transposed
        self._format = format
        self._auto_index = True # true in none index set manually
        self._rows = []
        self._precision = []
        if isinstance(precision, list):
            self._precision = precision
            # adjust length of precision array
            if len(self._precision) < len(self._columns):
                self._precision = self._precision + [self._precision[-1]] * (len(self._columns)-len(self._precision))
        else: # if precision is a number
            self._precision = [precision] * len(self._columns)

    @property
    def rows(self):
        return self._rows
    @property
    def columns(self):
        return self._columns
    @property
    def format(self):
        return self._format

    @format.setter
    def format(self, value):
        self._format = value

    def column(self, col):
        if isinstance(col,str):
            col = self._columns.index(col)
        return [row[col] for row in self._rows]

    def row(self, row):
        return self._rows[row]

    def cell(self, row, col):
        return self._rows[row][col]

    def append_from_other_table_printer(self, table_printer):
        for row in table_printer.rows:
            if all([c1 == c2 for c1,c2 in zip(self._columns, table_printer.columns)]):
                self.append(row)
            else:
                raise RuntimeError(f"TablePrinter: cannot append row if columns not match. Appended columns {table_printer.colums}, hosted columns {self._columns}")

    def append(self, row, index=None):
        self._rows.append(row)
        self._indexes.append(str(len(self._rows)))

        if index is not None:
            self._auto_index = False  # at least one index set manually
            self._indexes[-1] = index

    def write_file(self, filepath):
        with open(filepath, "w") as file:
            file.write(str(self))

    def __str__(self):
        data = [self._columns]
        if self._indexing:
            data = [["Index"] + self._columns]

        for row, i in zip(self._rows, self._indexes):
            r = []
            if self._indexing:
                r.append(i)
            for x, p in zip(row, self._precision):
                if isinstance(x, (int, float, complex)) and not isinstance(x, bool):
                    r.append(("{0:."+str(p)+"f}").format(x))
                else:
                    r.append(x)
            data.append(r)

        if self._transposed:
            data = transposition(data)
            print(data)

        title = self._title

        if self._format == self.FORMAT_LATEX:
            return write_latex_table(data, self._indexing)
        elif self._format == self.FORMAT_TXT: # txt

            return title + "\n" + write_adjusted_table(data)
        elif self._format ==  self.FORMAT_CSV:
            return write_csv_table(data)
