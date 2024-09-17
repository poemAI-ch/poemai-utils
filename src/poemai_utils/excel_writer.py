def write_excel(filename, data):
    from openpyxl import Workbook

    if not isinstance(data, list):
        raise ValueError("data must be a list")

    for row in data:
        if not isinstance(row, list):
            raise ValueError("data must be a list of lists")

    wb = Workbook()
    ws = wb.active

    column_widths = {}

    def convert_to_str(e):
        if e is None:
            return None
        if isinstance(e, str):
            return e
        if isinstance(e, float) and np.isnan(e):
            return None
        if isinstance(e, bool):
            return e
        return str(e)

    def format_cell_text(cell_text):
        if (
            isinstance(cell_text, str)
            and cell_text.startswith("http")
            and "://" in cell_text
        ):
            return f'=HYPERLINK("{cell_text}", "{cell_text}")'
        return convert_to_str(cell_text)

    for row_nr, row in enumerate(data):
        for col_nr, col in enumerate(row):
            cell_text = format_cell_text(col)
            ws.cell(row=row_nr + 1, column=col_nr + 1, value=cell_text)
            if cell_text is not None:
                column_widths[col_nr] = max(
                    column_widths.get(col_nr, 0), len(cell_text)
                )

    for col_nr, width in column_widths.items():
        if width < 10:
            width = 10
        elif width > 60:
            width = 60
        ws.column_dimensions[chr(65 + col_nr)].width = width

    wb.save(filename)
