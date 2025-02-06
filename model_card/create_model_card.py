import json


def df_to_interactive_html(df):
    records = df.fillna(0).to_dict('records')

    html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Data Explorer</title>
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #34495e;
            --accent-color: #3498db;
            --border-color: #e0e0e0;
            --hover-color: #f5f6fa;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.4;
            color: var(--primary-color);
            padding: 1rem;
            max-width: 1400px;
            margin: 0 auto;
        }

        .controls {
            background: white;
            padding: 1rem;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }

        #searchInput {
            width: 300px;
            padding: 0.5rem;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            margin-right: 1rem;
            font-size: 0.9rem;
        }

        .toggle-button {
            background: var(--accent-color);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
        }

        .toggle-button:hover {
            opacity: 0.9;
        }

        .column-toggles {
            display: none;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border-color);
        }

        .column-toggles.visible {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 0.5rem;
        }

        .column-selector {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
            font-size: 0.9rem;
        }

        th {
            background: var(--secondary-color);
            color: white;
            font-weight: 500;
            cursor: pointer;
            user-select: none;
            position: sticky;
            top: 0;
        }

        th:hover {
            background: var(--primary-color);
        }

        tr:hover td {
            background: var(--hover-color);
        }

        td {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 300px;
        }
    </style>
</head>
<body>
    <h1> Oda OCR Models </h1>
    <h4>Click on a cloumn to sort the table...</h4>
    <div class="controls">
        <input type="text" id="searchInput" placeholder="Search by model name...">
        <button class="toggle-button" onclick="toggleColumnSelectors()">Toggle Columns</button>
        <div id="columnSelectors" class="column-toggles"></div>
    </div>
    <table id="dataTable">
        <thead><tr id="headerRow"></tr></thead>
        <tbody></tbody>
    </table>

    <script>
        const csvData = {data};

        let sortColumn = '';
        let sortAsc = true;
        let visibleColumns = Object.keys(csvData[0]);
        let filteredData = [...csvData];

        function toggleColumnSelectors() {{
            const selectors = document.getElementById('columnSelectors');
            selectors.classList.toggle('visible');
        }}

        function renderTable() {{
            const headerRow = document.getElementById('headerRow');
            headerRow.innerHTML = '';

            visibleColumns.forEach(column => {{
                const th = document.createElement('th');
                th.textContent = column;
                th.onclick = () => sortData(column);
                headerRow.appendChild(th);
            }});

            const tbody = document.querySelector('tbody');
            tbody.innerHTML = '';

            filteredData.forEach(row => {{
                const tr = document.createElement('tr');
                visibleColumns.forEach(column => {{
                    const td = document.createElement('td');
                    td.textContent = row[column] ?? 'N/A';
                    tr.appendChild(td);
                }});
                tbody.appendChild(tr);
            }});
        }}

        function sortData(column) {
            if (sortColumn === column) {
                sortAsc = !sortAsc;
            } else {
                sortColumn = column;
                sortAsc = true;
            }
        
            filteredData.sort((a, b) => {
                const aVal = a[column] ?? '';
                const bVal = b[column] ?? '';
                return sortAsc ? 
                    (aVal < bVal ? 1 : -1) :
                    (aVal > bVal ? 1 : -1);
            });
        
            renderTable();
        }

        function setupColumnSelectors() {{
            const container = document.getElementById('columnSelectors');
            Object.keys(csvData[0]).forEach(column => {{
                const div = document.createElement('div');
                div.className = 'column-selector';

                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = `col-${{column}}`;
                checkbox.checked = true;
                checkbox.onchange = () => {{
                    if (checkbox.checked) {{
                        visibleColumns.push(column);
                    }} else {{
                        visibleColumns = visibleColumns.filter(col => col !== column);
                    }}
                    renderTable();
                }};

                const label = document.createElement('label');
                label.htmlFor = `col-${{column}}`;
                label.textContent = column;

                div.appendChild(checkbox);
                div.appendChild(label);
                container.appendChild(div);
            }});
        }}

        document.getElementById('searchInput').oninput = (e) => {{
            const searchTerm = e.target.value.toLowerCase();
            filteredData = csvData.filter(row => 
                row.model_name.toLowerCase().includes(searchTerm)
            );
            renderTable();
        }};

        setupColumnSelectors();
        renderTable();
    </script>
</body>
</html>"""

    return html_template.replace("{data}", json.dumps(records))

if __name__ == '__main__':
    import pandas as pd
    csv = pd.read_csv('models.csv')
    csv = csv[[x for x in csv if not 'train' in x]]
    html_content = df_to_interactive_html(csv)
    with open('model_card.html', 'w') as f:
        f.write(html_content)