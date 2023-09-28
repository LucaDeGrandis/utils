import numpy as np
from math import ceil
from copy import deepcopy



def get_table_dimensions(table):
    rows = set()
    cols = set()
    for cell in table:
        for _row in range(cell['row'], cell['row'] + cell['row span']):
            rows.add(_row)
        for _col in range(cell['column'], cell['column'] + cell['column span']):
            cols.add(_col)
    n_rows = max(rows) + 1
    n_cols = max(cols) + 1
    return n_rows, n_cols


def assert_dict_list_table(table, n_rows=None, n_cols=None):
    if n_rows is None and n_cols is None:
        n_rows, n_cols = get_table_dimensions(table)
    for cell in table:
        assert cell['row'] + cell['row span'] - 1 <= n_rows
        assert cell['column'] + cell['column span'] - 1 <= n_cols


def get_grid(n_rows, n_cols):
    return np.full((n_rows, n_cols), 0).tolist()


def add_cell_to_grid(grid, cell):
    for _row in range(cell['row'], cell['row'] + cell['row span']):
        for _col in range(cell['column'], cell['column'] + cell['column span']):
            assert not grid[_row][_cel]
            grid[_row][_cel] = 1


def add_cell_text_to_grid(grid, cell):
    for _row in range(cell['row'], cell['row'] + cell['row span']):
        for _col in range(cell['column'], cell['column'] + cell['column span']):
            assert grid[_row][_col] == 0
            if _row==cell['row'] and _col==cell['column']:
                grid[_row][_col] = cell['text']
            else:
                grid[_row][_col] = ''


def add_cell_position_to_grid(grid, cell):
    for _row in range(cell['row'], cell['row'] + cell['row span']):
        for _col in range(cell['column'], cell['column'] + cell['column span']):
            assert grid[_row][_col] == 0
            grid[_row][_col] = (cell['row'], cell['column'])


def get_html_table(table, n_rows):
    html_table = '<table>'
    table = sorted(table, key=lambda x: (x['row'], x['column']))
    for _row in range(n_rows):
        row = filter(lambda x: x['row']==_row, table)
        html_table += '<tr>'
        for cell in row:
            cell_start = '<th' if ('is_header' in cell and cell['is_header']) else '<td'
            cell_end = '</th>' if ('is_header' in cell and cell['is_header']) else '</td>'
            html_table += cell_start
            if cell["row span"]>1:
                html_table += f' rowspan={cell["row span"]}'
            if cell["column span"]>1:
                html_table += f' colspan={cell["column span"]}'
            html_table += '>'
            html_table += cell['text']
            html_table += cell_end
        html_table += '</tr>'
    html_table += '</table>'
    return html_table


def get_plain_table(table, n_rows, separator='\t'):
    plain_table = ''
    table = sorted(table, key=lambda x: (x['row'], x['column']))
    for _row in range(n_rows):
        row = filter(lambda x: x['row']==_row, table)
        for cell in row:
            plain_table += cell['text']
            plain_table += separator
        if _row != n_rows-1:
            plain_table += '\n'
    return plain_table


def get_column_lengths(table, n_rows, n_cols):
    col_lengths = []
    for _col in range(n_cols):
        col_lengths.append([])
        col_cells = filter(lambda x:_col in list(range(x['column'], x['column'] + x['column span'])), table)
        for cell in col_cells:
            if cell['column span']==1:
                col_lengths[-1].append(ceil(len(cell['text'])/cell['column span']))
    col_lengths = [max(x) for x in col_lengths]
    return col_lengths


def get_plain_table_with_horizontal_spacing(table, n_rows, n_cols, separator='\t'):
    grid = get_grid(n_rows, n_cols)
    for cell in table:
        add_cell_position_to_grid(grid, cell)
    col_lengths = get_column_lengths(table, n_rows, n_cols)
    col_lengths = [x + len(separator) for x in col_lengths]
    col_lengths_cumulative = [sum(col_lengths[:_i]) for _i in range(1, len(col_lengths)+1)]
    row_texts = []
    for _row in range(len(grid)):
        row_texts.append('')
        for _col in range(len(grid[0])):
            if grid[_row][_col]==0:
                cell_text = ''
            else:
                if (_row,_col)==grid[_row][_col]:
                    cells = list(filter(lambda x:x['row']==_row and x['column']==_col, table))
                    assert len(cells)==1
                    cell_text = cells[0]['text']
                else:
                    cell_text = ''
            row_texts[-1] += cell_text
            missing_spaces = col_lengths_cumulative[_col] - len(row_texts[-1]) - len(separator)
            if missing_spaces>=0:
                row_texts[-1] += ' '*missing_spaces + separator
    plain_table = '\n'.join(row_texts)
    
    return plain_table


def split_long_cell_to_rows(cell, max_len=100):
    if len(cell['text']) > max_len:
        text_split = cell['text'].split(' ')
        cell_texts = []
        temp_text = ''
        row_span = list(range(cell['row span']))
        while text_split:
            text = text_split.pop(0)
            if len(temp_text + ' ' + text) < max_len:
                temp_text += ' ' + text
            else:
                cell_texts.append(temp_text)
                temp_text = ''
        cell_texts.append(temp_text)
        existing_rows = []
        new_rows = []
        for _row, text in enumerate(cell_texts):
            new_cell = deepcopy(cell)
            new_cell['text'] = text
            new_cell['row'] += _row
            new_cell['row span'] = 1 if (_row==len(cell_texts)-1 or cell['row span']-_row-1<1) else cell['row span']-_row
            if _row in row_span:
                existing_rows.append(new_cell)
            else:
                new_rows.append(new_cell)
        return existing_rows, new_rows
    else:
        return [cell], []


def split_long_cell_to_new_row(cell, max_len=100):
    if len(cell['text']) > max_len:
        text_split = cell['text'].split(' ')
        temp_text = text_split.pop(0)
        row_span = list(range(cell['row span']))
        while text_split:
            text = text_split[0]
            if len(temp_text + ' ' + text) <= max_len:
                temp_text += ' ' + text_split.pop(0)
            else:
                break
        old_cell = deepcopy(cell)
        old_cell['text'] = temp_text
        old_cell['row span'] = 1

        require_new_row = cell['row span'] - 1 < 1

        new_cell = deepcopy(cell)
        new_cell['text'] = ' '.join(text_split)
        new_cell['row'] += + 1
        new_cell['row span'] = max(1, cell['row span'])

        return old_cell, new_cell, require_new_row
    else:
        return cell, None, False


def get_plain_table_with_vertical_spacing(table, n_rows, n_cols, separator='\t', max_len=100):
    table_copy = deepcopy(table)
    index2rows_map = {}
    _row = 0
    while True:
        row = list(filter(lambda x: x['row']==_row, table_copy))
        row = sorted(row, key=lambda x:x['column'])
        table_copy = list(filter(lambda x: x['row']!=_row, table_copy))
        row_cells = []
        new_cells = []
        new_row_requirements = []
        for cell in row:
            old_cell, new_cell, require_new_row = split_long_cell_to_new_row(cell, max_len)
            # print(old_cell)
            # print(new_cell)
            row_cells.append(old_cell)
            if new_cell is not None:
                new_cells.append(new_cell)
            new_row_requirements.append(require_new_row)
        for cell in row_cells:
            cell['row'] = _row
        if not sum(new_row_requirements):
            for cell in new_cells:
                cell['row span'] = max(1, cell['row span']-1)
                table_copy.append(cell)
        else:
            for cell in new_cells:
                cell['row span'] = max(1, cell['row span']-1)
            for cell in table_copy:
                cell['row'] += 1
            table_copy.extend(new_cells)
        index2rows_map[_row] = row_cells
        if not table_copy:
            break
        _row = min([x['row'] for x in table_copy])

    new_table = []
    for key, row in index2rows_map.items():
        new_table += row
    new_table = sorted(new_table,key=lambda x:(x['row'],x['column']))
    n_rows, n_cols = get_table_dimensions(new_table)

    return get_plain_table_with_horizontal_spacing(new_table, n_rows, n_cols, separator)


def get_json_table(table):
    assert table
    table = sorted(table, key=lambda x:(x['row'], x['column']))
    json_table = ''
    for entry in table:
        json_table += '{'
        json_table += f'"row": {entry["row"]}'
        json_table += f', "column": {entry["column"]}'
        if 'row span' in entry and entry['row span']>1:
            json_table += f', "row span": {entry["row span"]}'
        if 'column span' in entry and entry['column span']>1:
            json_table += f', "column span": {entry["column span"]}'
        json_table += f', "text": {entry["text"]}'
        if 'is_header' in entry and entry['is_header']:
            json_table += f', "is_header": True'
        json_table += '}, '
    json_table = json_table[:-2]
    return json_table
