import numpy as np
from math import ceil
from copy import deepcopy
from collections import defaultdict
from typing import List, Dict, Union, Any, Tuple



def table_rows_to_dict_list_without_row_column_index(
    table_rows: List[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Converts a list of lists representing the rows of a table to a list of dictionaries representing the cells of the table.

    Args:
        table_rows (list): A list of lists representing the rows of the table.
            Each row is a list of dictionaries representing the cells of the table.
            Each dictionary should have the following keys: 'text', 'row span', 'column span', and 'is_header'.
            The 'row span' and 'column span' keys represent the number of rows and columns that the cell spans over, respectively.
            The 'row span' and 'column span' keys are optional. If absent they are set to 1 by default.
            There can be extra rows at the end of the table but they are not expected between existing rows.

    Returns:
        list: A list of dictionaries representing the cells of the table.
            The dictionaries have the same keys as the input table.

    Raises:
        TypeError: If the input table is not a list of lists of dictionaries.
    """

    def update_grid(
        grid: List[List[int]]
    ) -> List[List[int]]:
        """Updates a grid to have the same number of columns in each row.

        Args:
            grid: A list of lists representing the grid. Each inner list represents a row of the grid and contains
                integers representing the cells in the row.

        Returns:
            None. The function modifies the input grid in place.

        Raises:
            None.
        """
        max_n_cols = max([len(x) for x in grid])
        for grid_row in grid:
            if len(grid_row) < max_n_cols:
                grid_row.extend([0]*(max_n_cols-len(grid_row)))

        return grid

    def check_grid_occupation(
        grid: List[List[int]],
        row: List[Dict[str, Any]]
    ) -> Tuple[bool, List[List[int]]]:
        """
        Checks if a given row of cells can be placed on a grid without overlapping with already occupied cells.

        Args:
            grid: A 2D list of integers representing the grid.
            row: A list of dictionaries representing the cells to be placed on the grid. Each dictionary contains the following keys:
                - 'column': An integer representing the starting column of the cell.
                - 'row': An integer representing the starting row of the cell.
                - 'column span': An integer representing the number of columns the cell spans.

        Returns:
            A tuple containing a boolean value indicating whether the cells can be placed on the grid without overlapping with already occupied cells, and a 2D list of integers representing the updated grid after placing the cells.
        """
        grid_c = deepcopy(grid)
        green_light = True
        for cell in row:
            cell_column = deepcopy(cell['column'])
            cell_row = deepcopy(cell['row'])
            while grid_c[cell_row][cell_column]:
                cell_column += 1
            for _col in range(cell_column, cell_column+cell['column span']):
                if _col >= len(grid_c[cell_row]):
                    green_light = False
                    break
                if not grid_c[cell_row][_col]:
                    grid_c[cell_row][_col] = 1
                else:
                    green_light = False
                    break

        return green_light, grid_c

    def update_table_rows(
        table_rows: List[List[Dict[str, Any]]],
        row_index: int
    ) -> List[List[Dict[str, Any]]]:
        """
        Updates the row index of all cells in the table that have a row index greater than or equal to the given row index.

        Args:
            table_rows: A list of lists of dictionaries representing the table rows and cells. Each dictionary contains the following keys:
                - 'row': An integer representing the row index of the cell.
                - 'column': An integer representing the column index of the cell.
                - Any other key-value pairs representing the cell data.
            row_index: An integer representing the row index from which the cells' row index should be updated.

        Returns:
            A list of lists of dictionaries representing the updated table rows and cells.
        """
        for row in table_rows:
            if not row:
                continue
            if row[0]['row'] < row_index:
                continue
            if row[0]['row'] >= row_index:
                for cell in row:
                    cell['row'] += 1

        return table_rows

    # def update_occupied_positions(
    #     occupied_positions: List[Tuple[int, int]],
    #     row_index: int
    # ) -> List[Tuple[int, int]]:
    #     for position in occupied_positions:
    #         if position[0] < row_index:
    #             continue
    #         if position[0] >= row_index:
    #             position[0] += 1

    #     return occupied_positions



    for row in table_rows:
        for cell in row:
            assert 'row' not in cell
            assert 'column' not in cell
            if 'column span' not in cell:
                cell['column span'] = 1
            if 'row span' not in cell:
                cell['row span'] = 1

    # Create and populate the table
    for _row, row in enumerate(table_rows):
        for _col, cell in enumerate(row):
            cell['row'] = _row
            cell['column'] = _col

    occupied_positions = []
    new_table = []
    grid = []
    row_pointer = -1
    while table_rows:
        row_pointer += 1
        row = table_rows.pop(0)
        grid.append([])
        grid_backup_1 = deepcopy(grid)
        row_occupied_positions = list(filter(lambda x: x[0]==row_pointer, occupied_positions))
        previous_n_cols = max([len(x) for x in grid])
        n_cols = max([previous_n_cols, sum([x['column span'] for x in row]) + len(row_occupied_positions)])
        grid[-1] = np.full(n_cols, 0).tolist()
        
        grid = update_grid(grid)
        
        for _row, _col in row_occupied_positions:
            grid[_row][_col] = 1
        
        # print('occupied_positions:', occupied_positions)
        # print('row_occupied_positions:', row_occupied_positions)
        # print('grid:', grid)

        grid_backup_2 = deepcopy(grid)
        grid_backup_1_size = max([len(x) for x in grid_backup_1])
        grid_backup_2 = [x[:grid_backup_1_size] for x in grid_backup_2]

        # Check if the current row cells can fit in the grid
        green_light, grid_c = check_grid_occupation(grid, row)
        # print('green_light:', green_light)
        if green_light:
            for cell in row:
                while grid[cell['row']][cell['column']]:
                    cell['column'] += 1
                for _row in range(cell['row'], cell['row']+cell['row span']):
                    for _col in range(cell['column'], cell['column']+cell['column span']):
                        if _row == cell['row']:
                            grid[cell['row']][_col] = 1
                        else:
                            occupied_positions.append([_row, _col])
                new_table.append(cell)
        else:
            table_rows.insert(0, row)
            update_table_rows(table_rows, row[0]['row'])
            # update_occupied_positions(occupied_positions, row[0]['row'])
            grid = grid_backup_2
            continue

    for _row in range(len(grid)):
        for _col in range(len(grid[_row])):
            if grid[_row][_col]:
                continue
            new_table.append({
                'row': _row,
                'column': _col,
                'row span': 1,
                'column span': 1,
                'is_header': False,
                'text': ''
            })

    new_table = sorted(new_table, key=lambda x: (x['row'], x['column']))

    return new_table


def complete_table(
    table: List[Dict[str, Any]],
    completion: str='',
) -> List[Dict[str, Any]]:
    """Adds empty cells to a table to complete it.

    Args:
        table: A list of dictionaries representing the cells of the table. Each dictionary should have the
            following keys: 'text', 'row', 'column', 'row span', 'column span', and 'is_header'. The 'row' and
            'column' keys represent the 0-based row and column indexes of the cell, respectively. The 'row span'
            and 'column span' keys represent the number of rows and columns that the cell spans over, respectively.
        completion: A string to use as the text for the empty cells that are added to the table. Defaults to ''.

    Returns:
        A new list of dictionaries representing the cells of the completed table. The dictionaries have the same
        keys as the input table.

    Raises:
        AssertionError: If the input table is not a list of dictionaries or if the table contains invalid cells.
    """

    # Populate the grid with non-empty spaces in the table
    n_rows, n_cols = get_table_dimensions(table)
    grid = get_grid(n_rows, n_cols)
    assert_dict_list_table(table)
    grid = assert_cell_position_uniqueness(table)

    # Introduce empty cells where the grid has empty spaces
    for _row, row in enumerate(grid):
        for _col, cell in enumerate(row):
            if not cell:
                table.append({'row': _row, 'column': _col, 'row span': 1, 'column span': 1, 'text': completion})

    return table


def correct_table_rows(
    table: List[Dict[str, Union[str, int, bool]]]
) -> List[List[Dict[str, Union[str, int, bool]]]]:
    """
    Corrects the row indexes of the cells in the given table.

    Args:
        table: A list of lists representing the rows of the table. Each row is a list of dictionaries
            representing the cells of the table. Each dictionary should have the following keys:
            'text', 'row', 'column', 'row span', 'column span', and 'is_header'. The 'row' and 'column'
            keys represent the 0-based row and column indexes of the cell, respectively. The 'row span'
            and 'column span' keys represent the number of rows and columns that the cell spans over,
            respectively.

    Returns:
        A new list of lists representing the corrected table. The dictionaries have the same keys as
        the input table.

    Raises:
        TypeError: If the input table is not a list of lists of dictionaries.
    """
    # Make a deep copy of the input table to avoid modifying it
    table_copy = deepcopy(table)

    # Isolate the rows:
    rows_indexes = set([i for cell in table_copy for i in range(cell['row'], cell['row']+cell['row span'])])
    rows_indexes = sorted(rows_indexes)
    
    # Put the cells in a row_dictionary
    row_dictionary = {}
    for row_index in rows_indexes:
        row = list(filter(lambda x: x['row']==row_index, table_copy))
        for cell in row:
            cell['priority'] = 1
        row_dictionary[row_index] = row

    # Loop over the rows of the table
    pointer = 0
    next_row_occupied = []
    new_table = []
    while pointer < len(rows_indexes):
        # add the next row occupied cells to the relative row
        for cell in next_row_occupied:
            row_dictionary[cell['row']].append(cell)

        # Update the pointer and get the current row
        i = pointer
        row = row_dictionary[i]
        occupying_cells = list(filter(lambda x:x['priority']==0, row))
        row = list(filter(lambda x:x['priority']==1, row))

        # Initialize a pointer to keep track of the occupied columns
        occupied_columns = [x for cell in occupying_cells for x in range(cell['column'], cell['column'] + cell['column span'])]

        # Loop over the cells in the current row
        new_row = []
        while row:
            cell = row.pop(0)
            while cell['column'] in occupied_columns:
                cell['column'] += 1
            cell_columns = list(range(cell['column'], cell['column'] + cell['column span']))
            assert set(cell_columns) & set(occupied_columns) == set()
            new_row.append(cell)
            occupied_columns.extend(list(range(cell['column'], cell['column'] + cell['column span'])))

        new_table.extend(list(filter(lambda x: x['priority']==1, new_row)))

        # Get the occupied positions in the next row
        next_row_occupied = []
        for cell in new_row:
            for j in range(cell['row']+1, cell['row']+cell['row span']):
                next_row_occupied.append({'row': j, 'column': cell['column'], 'row span': 1, 'column span': cell['column span'], 'priority': 0})

        # update the pointer
        pointer += 1

    return new_table


def get_table_dimensions(
    table: List[Dict[str, Any]]
) -> Tuple[int, int]:
    """
    Returns the number of rows and columns in a table.

    Args:
        table (list): A list of dictionaries representing cells in the table. Each dictionary contains the keys 'row', 'column', 'row span', and 'column span'.

    Returns:
        tuple: A tuple containing the number of rows and columns in the table.
    """
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


def assert_dict_list_table(
    table: List[Dict[str, Any]],
    n_rows: int=None,
    n_cols: int=None
) -> None:
    """
    Asserts that the cells in a table do not exceed the specified number of rows and columns.

    Args:
        table (list): A list of dictionaries representing cells in the table. Each dictionary contains the keys 'row', 'column', 'row span', and 'column span'.
        n_rows (int, optional): The maximum number of rows in the table. Defaults to None.
        n_cols (int, optional): The maximum number of columns in the table. Defaults to None.

    Raises:
        AssertionError: If any cell in the table exceeds the specified number of rows or columns.
    """
    if n_rows is None and n_cols is None:
        n_rows, n_cols = get_table_dimensions(table)
    for cell in table:
        assert cell['row'] + cell['row span'] - 1 <= n_rows
        assert cell['column'] + cell['column span'] - 1 <= n_cols


def get_grid(
    n_rows: int,
    n_cols: int
) -> List[List[int]]:
    """
    Returns a 2D list representing a grid with the specified number of rows and columns.

    Args:
        n_rows (int): The number of rows in the grid.
        n_cols (int): The number of columns in the grid.

    Returns:
        list: A 2D list representing the grid, with all elements initialized to 0.
    """
    return np.full((n_rows, n_cols), 0).tolist()


def add_cell_to_grid(
    grid: List[List[int]],
    cell: Dict[str, Any]
) -> None:
    """
    Adds a cell to a grid by setting the corresponding elements to 1.

    Args:
        grid (list): A 2D list representing the grid.
        cell (dict): A dictionary representing the cell to add. The dictionary must contain the keys 'row', 'column', 'row span', and 'column span'.

    Raises:
        AssertionError: If any element in the grid is already set to 1.
    """
    for _row in range(cell['row'], cell['row'] + cell['row span']):
        for _col in range(cell['column'], cell['column'] + cell['column span']):
            assert not grid[_row][_col]
            grid[_row][_col] = 1


def add_cell_text_to_grid(
    grid: List[List[str]],
    cell: Dict[str, Any]
) -> None:
    """
    Adds the text of a cell to a grid by setting the corresponding elements to the cell's text.

    Args:
        grid (list): A 2D list representing the grid.
        cell (dict): A dictionary representing the cell to add. The dictionary contains the keys 'row', 'column', 'row span', 'column span', and 'text'.

    Raises:
        AssertionError: If any element in the grid is already set to a non-empty string.
    """
    for _row in range(cell['row'], cell['row'] + cell['row span']):
        for _col in range(cell['column'], cell['column'] + cell['column span']):
            assert grid[_row][_col] == 0
            if _row==cell['row'] and _col==cell['column']:
                grid[_row][_col] = cell['text']
            else:
                grid[_row][_col] = ''


def add_cell_position_to_grid(
    grid: List[List[str]],
    cell: Dict[str, Any]
) -> None:
    """
    Adds the position of a cell to a grid by setting the corresponding elements to a tuple of the cell's row and column.

    Args:
        grid (list): A 2D list representing the grid.
        cell (dict): A dictionary representing the cell to add. The dictionary contains the keys 'row', 'column', 'row span', and 'column span'.

    Raises:
        AssertionError: If any element in the grid is already set to a non-zero tuple.
    """
    for _row in range(cell['row'], cell['row'] + cell['row span']):
        for _col in range(cell['column'], cell['column'] + cell['column span']):
            assert grid[_row][_col] == 0
            grid[_row][_col] = (cell['row'], cell['column'])


def assert_cell_position_uniqueness(
    table: List[Dict[str, Any]]
) -> None:
    """
    Asserts that each cell in a table has a unique position.

    Args:
        table (List[Dict[str, Any]]): A list of dictionaries representing cells in the table. Each dictionary must contain the keys 'row', 'column', 'row span', and 'column span'.

    Raises:
        AssertionError: If any two cells in the table have the same position.
    """
    n_rows, n_cols = get_table_dimensions(table)
    grid = get_grid(n_rows, n_cols)
    for cell in table:
        add_cell_to_grid(grid, cell)
    for cell in table:
        assert cell

    return grid


def get_html_table(
    table: List[Dict[str, Any]],
    n_rows: int
) -> str:
    """
    Returns an HTML table string from a list of dictionaries representing cells in the table.

    Args:
        table (list): A list of dictionaries representing cells in the table. Each dictionary contains the keys 'row', 'column', 'row span', 'column span', 'text', and 'is_header' (optional).
        n_rows (int): The number of rows in the table.

    Returns:
        str: An HTML table string.
    """
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


def get_html_table_with_head_and_body(
    table: List[Dict[str, Any]],
    n_rows: int,
    separator: str ='',
    force_thead: bool =False
) -> str:
    """
    Returns an HTML table string from a list of dictionaries representing cells in the table.

    Args:
        table (list): A list of dictionaries representing cells in the table. Each dictionary contains the keys 'row', 'column', 'row span', 'column span', 'text', and 'is_header' (optional).
        n_rows (int): The number of rows in the table.

    Returns:
        str: An HTML table string.
    """
    html_table = '<table>' + separator
    table = sorted(table, key=lambda x: (x['row'], x['column']))
    
    # Identify header rows
    headers = []
    bodies = []
    for _row in range(n_rows):
        row = list(filter(lambda x: x['row']==_row, table))
        if not force_thead:
            for cell in row:
                if 'is_header' in cell and cell['is_header']:
                    headers.append(_row)
                    break
                else:
                    bodies.append(_row)
        else:
            row = list(filter(lambda x: x['text'].strip()!='', row))
            row_head = list(filter(lambda x: 'is_header' in x and x['is_header'], row))
            row_body = list(filter(lambda x: ('is_header' not in x) or ('is_header' in x and not x['is_header']), row))
            if row_head and not row_body:
                headers.append(_row)
            else:
                bodies.append(_row)
    headers = sorted(headers)
    bodies = sorted(bodies)
    
    # Assert continuity of header rows
    if len(headers)>1:
        for _index in range(1, len(headers)):
            assert headers[_index] - headers[_index-1] == 1
    
    for _row in range(n_rows):
        row = list(filter(lambda x: x['row']==_row, table))
        
        # Assert that all cells are headers
        if _row in headers:
            for cell in row:
                if cell['text'].strip() != '':
                    assert 'is_header' in cell and cell['is_header']
        
        if _row == headers[0]:
            html_table += '<thead>' + separator
        elif _row == headers[-1] + 1:
            html_table += '<tbody>' + separator
        
        html_table += '<tr>' + separator
        
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
            html_table += cell_end + separator
        
        html_table += '</tr>' + separator
        
        if _row == headers[-1]:
            html_table += '</thead>' + separator
        elif _row == n_rows -1:
            html_table += '</tbody>' + separator

    html_table += '</table>'
    return html_table


def get_plain_table(
    table: List[Dict[str, Any]],
    n_rows: int,
    separator: str='\t',
    filter_empty: bool=False
) -> str:
    """
    Returns a plain text table string from a list of dictionaries representing cells in the table.

    Args:
        table (list): A list of dictionaries representing cells in the table. Each dictionary contains the keys 'row', 'column', 'row span', 'column span', and 'text'.
        n_rows (int): The number of rows in the table.
        separator (str, optional): The separator to use between cells. Defaults to '\t'.
        filter_empty (bool, optional): Whether to filter out empty cells or not. Default to False.

    Returns:
        str: A plain text table string.
    """
    plain_table = ''
    table = sorted(table, key=lambda x: (x['row'], x['column']))
    for _row in range(n_rows):
        if not filter_empty:
            row = filter(lambda x: x['row']==_row, table)
        else:
            row = filter(lambda x: x['row']==_row and x['text'].strip() != '', table)
        for cell in row:
            plain_table += cell['text']
            plain_table += separator
        if _row != n_rows-1:
            plain_table += '\n'
    return plain_table


def get_column_lengths(
    table: List[Dict[str, Any]],
    n_rows: int,
    n_cols: int
) -> List[int]:
    """
    Returns a list of the maximum lengths of each column in a table.

    Args:
        table (list): A list of dictionaries representing cells in the table. Each dictionary contains the keys 'row', 'column', 'row span', 'column span', and 'text'.
        n_rows (int): The number of rows in the table.
        n_cols (int): The number of columns in the table.

    Returns:
        list: A list of integers representing the maximum length of each column in the table.
    """
    col_lengths = []
    for _col in range(n_cols):
        col_lengths.append([])
        col_cells = filter(lambda x:_col in list(range(x['column'], x['column'] + x['column span'])), table)
        for cell in col_cells:
            col_lengths[-1].append(ceil(len(cell['text'])/cell['column span']))
    col_lengths = [max(x) for x in col_lengths]
    return col_lengths


def get_plain_table_with_horizontal_spacing(
    table: List[Dict[str, Any]],
    n_rows: int,
    n_cols: int,
    separator: str='\t'
) -> str:
    """
    Returns a plain text table string with horizontal spacing from a list of dictionaries representing cells in the table.

    Args:
        table (list): A list of dictionaries representing cells in the table. Each dictionary contains the keys 'row', 'column', 'row span', 'column span', and 'text'.
        n_rows (int): The number of rows in the table.
        n_cols (int): The number of columns in the table.
        separator (str, optional): The separator to use between cells. Defaults to '\t'.

    Returns:
        str: A plain text table string with horizontal spacing.
    """
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


def split_long_cell_to_rows(
    cell: Dict[str, Any],
    max_len: int=100
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits a cell into multiple rows if its text length exceeds a maximum length.

    Args:
        cell (dict): A dictionary representing a cell in the table. It contains the keys 'row', 'column', 'row span', 'column span', and 'text'.
        max_len (int, optional): The maximum length of the cell text. Defaults to 100.

    Returns:
        tuple: A tuple of two lists. The first list contains the existing rows of the cell, and the second list contains the new rows created from splitting the cell.
    """
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


def split_long_cell_to_new_row(
    cell: Dict[str, Any],
    max_len: int=100
):
    """
    Splits a cell into two cells if its text length exceeds a maximum length.

    Args:
        cell (dict): A dictionary representing a cell in the table. It contains the keys 'row', 'column', 'row span', 'column span', and 'text'.
        max_len (int, optional): The maximum length of the cell text. Defaults to 100.

    Returns:
        tuple: A tuple of three elements. The first element is the old cell with the text truncated to fit the maximum length. The second element is the new cell with the remaining text. The third element is a boolean indicating whether a new row is required.
    """
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


def get_plain_table_with_vertical_spacing(
    table: List[Dict[str, Any]],
    n_rows: int,
    n_cols: int,
    separator: str='\t',
    max_len: int=100
):
    """
    Returns a plain text table string with vertical spacing from a list of dictionaries representing cells in the table.

    Args:
        table (list): A list of dictionaries representing cells in the table. Each dictionary contains the keys 'row', 'column', 'row span', 'column span', and 'text'.
        n_rows (int): The number of rows in the table.
        n_cols (int): The number of columns in the table.
        separator (str, optional): The separator to use between cells. Defaults to '\t'.
        max_len (int, optional): The maximum length of the cell text before it is split into multiple rows. Defaults to 100.

    Returns:
        str: A plain text table string with vertical spacing.
    """
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


def get_json_table(
    table: List[Dict[str, Any]]
):
    """
    Returns a JSON string representation of a table.

    Args:
        table (list): A list of dictionaries representing cells in the table. Each dictionary contains the keys 'row', 'column', 'row span', 'column span', 'text', and 'is_header' (optional).

    Returns:
        str: A JSON string representation of the table.
    """
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
