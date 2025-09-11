import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State, ALL
import pandas as pd
import numpy as np
import base64, io, ast
import plotly.express as px
import plotly.io as pio
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from flask import Flask, session, Response
import hashlib
import sqlite3
import pickle
import time

server = Flask(__name__)
server.secret_key = os.environ.get("DASH_SECRET_KEY", "dev_secret_key")

@server.route('/style.css')
def serve_css():
    css_path = os.path.join(os.path.dirname(__file__), 'style.css')
    with open(css_path) as f:
        css_content = f.read()
    return Response(css_content, mimetype='text/css')

app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True)
app.title = "Dashboard"

# Automatic cache-busting with timestamp
timestamp = int(time.time())


app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <link rel="stylesheet" type="text/css" href="/style.css?v={timestamp}">
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''

conn = sqlite3.connect('users.db')
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstname TEXT NOT NULL,
    lastname TEXT NOT NULL,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
''')
c.execute('''
CREATE TABLE IF NOT EXISTS datasets (
    username TEXT PRIMARY KEY,
    data_blob BLOB
)
''')
conn.commit()
conn.close()

login_layout = html.Div([
    html.Div([
        html.H2("Login"),
        dcc.Input(id='login-username', type='text', placeholder='Username', className="auth-input"),
        dcc.Input(id='login-password', type='password', placeholder='Password', className="auth-input"),
        html.Button("Login", id='login-btn', className="auth-button"),
        html.Div(id='login-msg'),
        html.A("Go to Register", href='/register', className="auth-link")
    ], className="auth-container")
])

register_layout = html.Div([
    html.Div([
        html.H2("Register"),
        dcc.Input(id='reg-firstname', type='text', placeholder='First Name', className="auth-input"),
        dcc.Input(id='reg-lastname', type='text', placeholder='Last Name', className="auth-input"),
        dcc.Input(id='reg-username', type='text', placeholder='Username', className="auth-input"),
        dcc.Input(id='reg-password', type='password', placeholder='Password', className="auth-input"),
        html.Button("Register", id='register-btn', className="auth-button"),
        html.Div(id='register-msg'),
        html.A("Go to Login", href='/login', className="auth-link")
    ], className="auth-container")
])

def dashboard_layout(username):
    return html.Div([
        html.Div([
            html.H1(f"Welcome, {username}", style={'display':'inline-block'}),
            html.Button("Logout", id={'type':'logout-btn','index':0}, style={'float':'right'})
        ], style={'padding':'20px', 'borderBottom':'2px solid #ccc'}),
        dcc.Store(id='stored-data', storage_type='session'),
        dcc.Store(id='feature-store', storage_type='session'),
        dcc.Store(id='selected-columns', storage_type='session'),
        html.Div(id='hidden-results', style={'display':'none'}),
        dcc.Tabs(id="tabs", value='tab-data', children=[
            dcc.Tab(label='Data', value='tab-data'),
            dcc.Tab(label='Visualization', value='tab-viz'),
            dcc.Tab(label='ML Prediction', value='tab-ml'),
            dcc.Tab(label='Data Profiling', value='tab-profile'),
            dcc.Tab(label='Reports', value='tab-report')
        ]),
        html.Div(id='tabs-content', style={'padding':'20px'})
    ])


app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    html.Div(id='page-content')
])


@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'), Input({'type':'logout-btn','index':ALL}, 'n_clicks')]
)
def display_page(pathname, logout_clicks):
    ctx = dash.callback_context
    if ctx.triggered and 'logout-btn' in ctx.triggered[0]['prop_id']:
        session.pop('username', None)
        return login_layout
    if 'username' in session:
        return dashboard_layout(session['username'])
    if pathname == '/register':
        return register_layout
    return login_layout

@app.callback(
    Output('register-msg', 'children'),
    Input('register-btn', 'n_clicks'),
    State('reg-firstname', 'value'),
    State('reg-lastname', 'value'),
    State('reg-username', 'value'),
    State('reg-password', 'value'),
    prevent_initial_call=True
)
def register_user(n_clicks, firstname, lastname, username, password):
    if not all([firstname, lastname, username, password]):
        return "Please fill in all fields."
    hashed = hashlib.sha256(password.encode()).hexdigest()
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (firstname, lastname, username, password) VALUES (?, ?, ?, ?)",
                  (firstname, lastname, username, hashed))
        conn.commit()
        conn.close()
        return "Registered successfully! Go to login."
    except sqlite3.IntegrityError:
        return "Username already exists."

@app.callback(
    [Output('url', 'pathname'), Output('login-msg', 'children')],
    Input('login-btn', 'n_clicks'),
    State('login-username', 'value'),
    State('login-password', 'value'),
    prevent_initial_call=True
)
def login_user(n_clicks, username, password):
    if n_clicks is None or not username or not password:
        return dash.no_update, "Enter username and password."
    hashed = hashlib.sha256(password.encode()).hexdigest()
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed))
    user = c.fetchone()
    conn.close()
    if user:
        session['username'] = username
        return '/', ""
    else:
        return dash.no_update, "Invalid username or password."


def parse_contents(contents, filename):
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if 'csv' in filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename.lower() or 'xlsx' in filename.lower():
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None
        return df
    except Exception as e:
        print(f"Error parsing file: {e}")
        return None

def get_df_from_db():
    if 'username' in session:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT data_blob FROM datasets WHERE username=?", (session['username'],))
        data = c.fetchone()
        conn.close()
        if data and data[0]:
            return pd.read_pickle(io.BytesIO(data[0]))
    return None

def preprocess_df(df, missing_option, scaling_option, x_cols):
    df = df.copy()
    if missing_option=="drop":
        df = df.dropna()
    elif missing_option=="mean":
        df = df.fillna(df.mean(numeric_only=True))
    elif missing_option=="median":
        df = df.fillna(df.median(numeric_only=True))
    numeric_x_cols = df[x_cols].select_dtypes(include=np.number).columns.tolist()
    if numeric_x_cols:
        if scaling_option=="standard":
            scaler = StandardScaler()
            df[numeric_x_cols] = scaler.fit_transform(df[numeric_x_cols])
        elif scaling_option=="minmax":
            scaler = MinMaxScaler()
            df[numeric_x_cols] = scaler.fit_transform(df[numeric_x_cols])
    return df


@app.callback(
    Output('tabs-content','children'),
    Input('tabs','value')
)
def render_content(tab):
    if tab=='tab-data':
        return html.Div([
            html.H3("Upload your dataset"),
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag and Drop or ', html.A('Select CSV or Excel File')]),
                style={'width':'50%','height':'60px','lineHeight':'60px', 'borderWidth':'2px','borderStyle':'dashed','borderRadius':'10px', 'textAlign':'center','margin':'20px auto'},
                multiple=False
            ),
            html.Button("Clear Dataset / Upload new", id="clear-dataset-btn", style={'margin':'10px'}),
            html.Div(id='clear-confirm', style={'color':'red','margin':'10px'}),
            html.Div(id='output-table')
        ])
    elif tab=='tab-viz':
        return html.Div([
            html.Div([
                html.Div([
                    html.H4("Controls"),
                    html.Label("Select Feature Columns (X):"),
                    dcc.Dropdown(id='x-axis', multi=True),
                    html.Label("Select Target Columns (Y):"),
                    dcc.Dropdown(id='y-axis', multi=True),
                    html.Label("Select Chart Type:"),
                    dcc.Dropdown(
                        id='chart-type',
                        options=[
                            {"label":"Scatter","value":"scatter"},
                            {"label":"Line","value":"line"},
                            {"label":"Bar","value":"bar"},
                            {"label":"Histogram","value":"histogram"},
                            {"label":"Box","value":"box"},
                            {"label":"Pie","value":"pie"},
                            {"label":"Scatter Matrix","value":"scatter_matrix"}
                        ],
                        value="scatter"
                    ),
                    html.Div(id='chart-suggestion', style={'marginTop':'10px','fontWeight':'bold'})
                ], style={'width':'30%','display':'inline-block','verticalAlign':'top','padding':'10px','borderRight':'1px solid #ccc'}),
                html.Div([
                    dcc.Graph(id='graph'),
                    dcc.Graph(id='pair-heatmap')
                ], style={'width':'68%','display':'inline-block','padding':'10px'})
            ])
        ])
    elif tab=='tab-ml':
        return html.Div([
            html.Div([
                html.H4("ML Settings"),
                html.Label("Missing value handling:"),
                dcc.Dropdown(
                    id='missing-option',
                    options=[
                        {"label":"Drop rows","value":"drop"},
                        {"label":"Fill with Mean","value":"mean"},
                        {"label":"Fill with Median","value":"median"}
                    ],
                    value="drop"
                ),
                html.Label("Feature scaling:"),
                dcc.Dropdown(
                    id='scaling-option',
                    options=[
                        {"label":"None","value":"none"},
                        {"label":"StandardScaler","value":"standard"},
                        {"label":"MinMaxScaler","value":"minmax"}
                    ],
                    value="none"
                ),
                html.Label("Select ML Model:"),
                dcc.Dropdown(
                    id='ml-model',
                    options=[
                        {"label":"Linear Regression","value":"linear"},
                        {"label":"Decision Tree","value":"tree"},
                        {"label":"Random Forest","value":"forest"},
                        {"label":"Gradient Boosting","value":"gb"}
                    ],
                    value="linear"
                ),
                html.Div(id='input-fields', style={'marginTop':'20px'})
            ], style={'width':'30%','display':'inline-block','verticalAlign':'top','padding':'10px','borderRight':'1px solid #ccc'}),
            html.Div([
                html.H4("Prediction & Evaluation"),
                html.Div(id='ml-prediction', style={'marginTop':'20px','fontWeight':'bold','color':'green'}),
                html.Div(id='ml-evaluation', style={'marginTop':'10px','fontWeight':'bold','color':'blue'})
            ], style={'width':'68%','display':'inline-block','padding':'10px'})
        ])
    elif tab=='tab-profile':
        return html.Div([
            html.H3("Data Profiling"),
            html.Div(id='profile-output')
        ])
    elif tab=='tab-report':
        return html.Div([
            html.H3("Reports"),
            html.Div([
                html.H5("Select sections to include:"),
                dcc.Checklist(
                    id='report-sections-checklist',
                    options=[
                        {'label': 'Data Summary', 'value': 'data_summary'},
                        {'label': 'Prediction & Evaluation', 'value': 'eval_metrics'},
                        {'label': 'Feature Importance', 'value': 'feature_importance'},
                        {'label': 'Example Graph', 'value': 'graph_plot'}
                    ],
                    value=['data_summary', 'eval_metrics'],
                    inline=False
                ),
                html.Button("Download PDF Report", id="download-pdf-btn"),
                dcc.Download(id="download-pdf"),
            ], style={'textAlign':'center','margin':'20px'})
        ])


@app.callback(
    [Output('output-table','children'), Output('stored-data','data'), Output('clear-confirm','children')],
    [Input('upload-data','contents'), Input('clear-dataset-btn','n_clicks')],
    [State('upload-data','filename')],
    prevent_initial_call=True
)
def store_csv_in_db(contents, clear_click, filename):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if 'username' not in session:
        return None, None, "Please log in."
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    if trigger_id == "clear-dataset-btn":
        c.execute("DELETE FROM datasets WHERE username=?", (session['username'],))
        conn.commit()
        conn.close()
        return "Upload a file.", None, "Dataset cleared. You can upload a new one."
    if contents:
        df = parse_contents(contents, filename)
        if df is not None:
            buffer = io.BytesIO()
            df.to_pickle(buffer)
            data_blob = buffer.getvalue()
            c.execute("INSERT OR REPLACE INTO datasets (username, data_blob) VALUES (?, ?)", (session['username'], data_blob))
            conn.commit()
            table = dash_table.DataTable(
                data=df.head(10).to_dict('records'),
                columns=[{"name":i,"id":i} for i in df.columns],
                page_size=5,
                style_table={'overflowX':'auto'}
            )
            conn.close()
            return table, 'data_loaded', ""
    conn.close()
    return "Upload a file.", None, ""

@app.callback(
    Output('selected-columns','data'),
    [Input('x-axis','value'), Input('y-axis','value')]
)
def update_selected_columns(x_cols, y_cols):
    return {"x": x_cols, "y": y_cols}

@app.callback(
    [Output('x-axis','options'), Output('y-axis','options')],
    [Input('stored-data','data')]
)
def set_dropdowns(data_loaded_flag):
    df = get_df_from_db()
    if df is not None:
        opts = [{"label":col,"value":col} for col in df.columns]
        return opts, opts
    return [], []

@app.callback(
    [Output('graph','figure'), Output('pair-heatmap','figure'), Output('chart-suggestion','children')],
    [Input('selected-columns','data'), Input('chart-type','value'), Input('stored-data','data')]
)
def update_graph(selected_columns, chart_type, data_loaded_flag):
    df = get_df_from_db()
    x_cols = selected_columns.get('x', []) if selected_columns else []
    y_cols = selected_columns.get('y', []) if selected_columns else []
    empty_fig = {"layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False}}}

    if df is None:
        return empty_fig, empty_fig, "Please upload a dataset to begin."
    if not x_cols and chart_type not in ['histogram']:
        return empty_fig, empty_fig, "Select at least one X-axis column."
    if not y_cols and chart_type in ['scatter', 'line', 'bar', 'box', 'pie']:
        return empty_fig, empty_fig, f"Select a Y-axis column for a {chart_type} chart."

    suggestion = ""
    fig = empty_fig
    try:
        numeric_x = all(np.issubdtype(df[col].dtype, np.number) for col in x_cols)
        numeric_y = all(np.issubdtype(df[col].dtype, np.number) for col in y_cols)

        if numeric_x and numeric_y:
            suggestion = "Suggestion: Scatter or Line plot for numeric relationships."
        elif not numeric_x and numeric_y:
            suggestion = "Suggestion: Bar or Box plot for categorical data."

        if chart_type == 'scatter_matrix' and len(x_cols) >= 2:
            fig = px.scatter_matrix(df, dimensions=x_cols)
        elif chart_type == 'pie' and y_cols and x_cols:
            if np.issubdtype(df[y_cols[0]].dtype, np.number):
                fig = px.pie(df, values=y_cols[0], names=x_cols[0], title=f"Pie Chart of {y_cols[0]} by {x_cols[0]}")
            else:
                suggestion = "Error: Pie chart requires a numeric Y-axis (values)."
        elif chart_type == 'histogram' and x_cols:
            fig = px.histogram(df, x=x_cols[0])
        elif x_cols and y_cols:
            chart_map = {
                "scatter": px.scatter,
                "line": px.line,
                "bar": px.bar,
                "box": px.box
            }
            fig_func = chart_map.get(chart_type, px.scatter)
            fig = fig_func(df, x=x_cols[0], y=y_cols[0])
    except Exception as e:
        suggestion = f"Graphing Error: {e}"

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        heatmap = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis', title="Correlation Heatmap")
    else:
        heatmap = {"layout": {"title": "Not enough numeric columns for a heatmap."}}

    return fig, heatmap, suggestion

@app.callback(
    Output('profile-output', 'children'),
    Input('tabs', 'value')
)
def generate_profiling_report(tab):
    if tab != 'tab-profile':
        return ""
    df = get_df_from_db()
    if df is None:
        return html.P("Please upload a dataset on the 'Data' tab first.")

    summary = df.describe(include='all').T.reset_index().round(3)
    summary.rename(columns={'index': 'Column'}, inplace=True)

    dtypes = pd.DataFrame(df.dtypes, columns=['Data Type']).reset_index()
    dtypes.rename(columns={'index': 'Column'}, inplace=True)
    dtypes['Data Type'] = dtypes['Data Type'].astype(str)

    return [
        html.H4("Summary Statistics"),
        dash_table.DataTable(
            data=summary.to_dict('records'),
            columns=[{"name": i, "id": i} for i in summary.columns],
            style_table={'overflowX': 'auto'}
        ),
        html.H4("Data Types & Missing Values"),
        dash_table.DataTable(
            data=df.isnull().sum().reset_index(name='missing_count').to_dict('records'),
            columns=[{"name": "Column", "id": "index"}, {"name": "Missing Count", "id": "missing_count"}],
            style_table={'overflowX': 'auto'}
        )
    ]


@app.callback(
    Output('input-fields','children'),
    Input('selected-columns','data'),
    State('stored-data', 'data')
)
def generate_inputs(selected_columns, data_loaded):
    if not data_loaded:
        return "Select features (X) to generate prediction inputs."
    df = get_df_from_db()
    if df is None:
        return ""
    features = selected_columns.get('x',[]) if selected_columns else []
    if features:
        inputs=[]
        numeric_features = df[features].select_dtypes(include=np.number).columns.tolist()
        for f in numeric_features:
            min_val, max_val = df[f].min(), df[f].max()
            inputs.append(html.Label(f"{f}:"))
            inputs.append(dcc.Slider(
                id={'type':'feature-input','index':f},
                min=min_val,
                max=max_val,
                step=(max_val-min_val)/100,
                value=df[f].mean(),
                tooltip={"placement":"bottom","always_visible":True}
            ))
        if len(numeric_features) > 0:
            inputs.append(html.Button("Predict", id={'type':'predict-btn','index':0}, n_clicks=0, style={'marginTop':'15px'}))
        return html.Div(inputs)
    else:
        return "Please select at least one numeric feature (X) to enable prediction."


@app.callback(
    Output('feature-store','data'),
    Input({'type':'feature-input','index':ALL}, 'value')
)
def store_feature_values(values):
    return values


@app.callback(
    [Output('ml-prediction','children'), Output('ml-evaluation','children'), Output('hidden-results','children')],
    [Input({'type':'predict-btn','index':ALL}, 'n_clicks')],
    [State('selected-columns','data'),
     State('ml-model','value'),
     State('stored-data','data'),
     State('missing-option','value'),
     State('scaling-option','value'),
     State({'type': 'feature-input', 'index': ALL}, 'id'),
     State({'type': 'feature-input', 'index': ALL}, 'value')],
    prevent_initial_call=True
)
def ml_predict(n_clicks_list, selected_columns, model_type, data_loaded_flag, missing_opt, scaling_opt, feature_ids, feature_values):
    if not n_clicks_list or sum(n_clicks_list) == 0:
        return "", "", ""
    if model_type is None or missing_opt is None or scaling_opt is None:
        return "Please select all ML settings.", "", ""
    df = get_df_from_db()
    if df is None or selected_columns is None or not selected_columns.get('x', []) or not selected_columns.get('y', []):
        return "Upload data and select both X and Y columns first.", "", ""

    x_cols = selected_columns.get('x',[])
    y_cols = selected_columns.get('y',[])
    numeric_x = df[x_cols].select_dtypes(include=np.number).columns.tolist()
    numeric_y = df[y_cols].select_dtypes(include=np.number).columns.tolist()
    if not numeric_x or not numeric_y:
        return "Prediction requires at least one numeric X and one numeric Y column.", "", ""

    try:
        input_feature_names = [fid['index'] for fid in feature_ids]
        if not feature_values or len(feature_values) != len(input_feature_names):
            return "Please input a value for each feature and click predict.", "", ""

        df_processed = preprocess_df(df.copy(), missing_opt, scaling_opt, numeric_x)
        if df_processed.empty:
            return "Error: Dataset is empty after preprocessing. 'Drop rows' may have removed all data.", "", ""

        results = {}
        pred_texts = []
        eval_texts = []

        for y_col in numeric_y:
            y = df_processed[y_col].values
            X = df_processed[numeric_x].values

            model_dict = {
                "linear": LinearRegression(),
                "tree": DecisionTreeRegressor(random_state=42),
                "forest": RandomForestRegressor(n_estimators=50, random_state=42),
                "gb": GradientBoostingRegressor(n_estimators=50, random_state=42)
            }
            model = model_dict.get(model_type, LinearRegression())
            model.fit(X, y)

            input_data = np.array(feature_values).reshape(1, -1)
            pred_val = model.predict(input_data)[0]

            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)

            pred_texts.append(f"Prediction for '{y_col}': {pred_val:.3f}")
            eval_texts.append(f"Evaluation for '{y_col}': R²={r2:.3f}, MSE={mse:.3f}, MAE={mae:.3f}")
            results[y_col] = {"prediction": pred_texts[-1], "evaluation": eval_texts[-1], "model_type": model_type}

        return html.Pre("\n".join(pred_texts)), html.Pre("\n".join(eval_texts)), str(results)
    except Exception as e:
        return f"An error occurred: {e}", "", ""


@app.callback(
    Output("download-pdf","data"),
    Input("download-pdf-btn","n_clicks"),
    [State('hidden-results','children'),
     State('stored-data','data'),
     State('selected-columns','data'),
     State('report-sections-checklist', 'value')],
    prevent_initial_call=True
)
def download_pdf(n_clicks, hidden_results, data_loaded_flag, selected_columns, selected_sections):
    if not n_clicks:
        return None
    if not hidden_results or not data_loaded_flag or not selected_columns or not selected_sections:
        return None
    try:
        results = ast.literal_eval(hidden_results)
        df = get_df_from_db()
        if df is None:
            return None

        y_col_key = next(iter(results))
        model_type = results[y_col_key]['model_type']
        x_cols = selected_columns.get('x',[])
        y_cols = selected_columns.get('y',[])
        if not y_cols:
            return None

        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter
        y_position = height - 50

        def draw_multiline_text(canvas_obj, x, y, text):
            text_object = canvas_obj.beginText(x, y)
            for line in text.split('\n'):
                text_object.textLine(line)
            canvas_obj.drawText(text_object)
            return text_object.getY() - 15

        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y_position, f"Report")
        y_position -= 30

        if 'data_summary' in selected_sections:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_position, "Data Summary")
            y_position -= 20
            c.setFont("Helvetica", 12)
            stats_text = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
            y_position = draw_multiline_text(c, 50, y_position, stats_text)
            y_position -= 15

        if 'eval_metrics' in selected_sections:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_position, "Prediction & Evaluation")
            y_position -= 20
            c.setFont("Helvetica", 12)
            full_eval_text = "\n".join([res['evaluation'] for res in results.values()])
            y_position = draw_multiline_text(c, 50, y_position, full_eval_text)
            y_position -= 15

        numeric_x = df[x_cols].select_dtypes(include=np.number).columns.tolist()
        numeric_y = df[y_cols].select_dtypes(include=np.number).columns.tolist()

        if 'feature_importance' in selected_sections and model_type in ['tree', 'forest', 'gb'] and numeric_x and numeric_y:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_position, "Feature Importance")
            y_position -= 20
            c.setFont("Helvetica", 12)
            y_col_for_importance = numeric_y[0]
            X = df[numeric_x].values
            y = df[y_col_for_importance].values
            model = {"tree":DecisionTreeRegressor(), "forest":RandomForestRegressor(), "gb":GradientBoostingRegressor()}[model_type]
            model.fit(X, y)
            importances = "\n".join([f"{f}: {imp:.3f}" for f, imp in zip(numeric_x, model.feature_importances_)])
            y_position = draw_multiline_text(c, 50, y_position, importances)
            y_position -= 15

        if 'graph_plot' in selected_sections and x_cols and y_cols:
            if y_position < 350:
                c.showPage()
                y_position = height - 50
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_position, "Visualization")
            y_position -= 20
            fig = px.scatter(df, x=x_cols[0], y=y_cols[0], title=f"{y_cols[0]} vs. {x_cols[0]}")
            img_buffer = io.BytesIO()
            pio.write_image(fig, img_buffer, format='png', width=500, height=300)
            img_buffer.seek(0)
            img = ImageReader(img_buffer)
            c.drawImage(img, 50, y_position - 300)

        c.save()
        pdf_buffer.seek(0)
        return dcc.send_bytes(pdf_buffer.getvalue(), filename="ML_Report.pdf")
    except Exception as e:
        print(f"An error occurred during PDF generation: {e}")
        return None

if name == "__main__":
  run(Debug=True)
