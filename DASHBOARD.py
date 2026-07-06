import glob
import json
import dash
from dash import dcc, html, Input, Output
import geopandas as gpd
import plotly.express as px
import pandas as pd
import joblib


# 1. CARGA DE DATOS
# PANAMÁ
gdf = gpd.read_file("geoBoundaries-PAN-ADM2_simplified.geojson")
archivo_inec = glob.glob("datos_sociodemográfica_*.csv")[0]
df_inec = pd.read_csv(archivo_inec)

gdf["distrito_clean"] = gdf["shapeName"].str.strip().str.upper().str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
df_inec["distrito_clean"] = df_inec["Nombre Distrito"].str.strip().str.upper().str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
gdf_merged = gdf.merge(df_inec, on="distrito_clean", how="left").set_index("distrito_clean")
geojson_dict = json.loads(gdf_merged.to_json())

# RIESGO CARDIOVASCULAR (datos CSV)
df = pd.read_csv("cardiovascular_risk_dataset.csv",sep=';')
df =df.drop(['Patient_ID'],axis='columns')

modelo_real_svm = joblib.load('modelo_svm_clasificacion.pkl')
escalador_real = joblib.load('escalador_svm.pkl')

# 2. CONFIGURACIÓN DE LA APP Y LAYOUT
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Análisis de datos 2026"

ESTILOS = {
    'fondo': '#F0F3F4',
    'panel': '#FFFFFF',
    'azul': '#1B4F72',
    'azul2': '#2980B9',
    'teal': '#148F77',
    'borde': '1px solid #D5D8DC',
}

app.layout = html.Div(style={"backgroundColor": "#110C0C", "color": "#ffffff", "minHeight": "100vh", "fontFamily": "Segoe UI"}, children=[
    
    # Encabezado principal
    html.Div(style={"padding": "20px", "textAlign": "center", "backgroundColor": "#1f1f1f", "borderBottom": "2px solid #00fff0"}, children=[
        html.H1("Dashboard de Conectividad & Salud", style={"margin": "0", "color": "#00fff0"}),
        html.P("Proyecto Final", style={"opacity": "0.7"})
    ]),

    # Control de Pestañas
    dcc.Tabs(id="tabs-proyecto", value='tab-conectividad', children=[
        dcc.Tab(label='RIESGO CARDIOVASCULAR (SALUD)', value='tab-salud', style={"backgroundColor": "#1e1e2f"}, selected_style={"backgroundColor": "#00fff0", "color": "black"}),
        dcc.Tab(label='PREDICCIÓN DE RIESGO (MODELO)', value='tab-prediccion',style={"backgroundColor": "#1e1e2f"},selected_style={"background":"#00fff0","color":"black"}),
        dcc.Tab(label='Mapa de Panamá', value='tab-conectividad', style={"backgroundColor": "#1e1e2f"}, selected_style={"backgroundColor": "#00fff0", "color": "black"}),
        
    ]),

    # Contenedor dinámico que se llena según la pestaña activa
    html.Div(id='tabs-content')
])

# 3. CALLBACK DE NAVEGACIÓN Y RENDERIZADO DE GRÁFICOS
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs-proyecto', 'value')]
)
def render_content(tab):
    
    # Si el usuario hace clic en Salud, muestra ÚNICAMENTE tu gráfica del notebook
    
    if tab == 'tab-salud':
        
        # 1. GRÁFICA DE BARRAS (Tu Countplot original)
        fig_countplot = px.histogram(
            df, 
            x='risk_category', 
            title="Distribución del Riesgo (Medio, alto, bajo)", 
            template="plotly_white", 
            color_discrete_sequence=['#2980B9'],
            category_orders={'risk_category': ["Medium", "High", "Low"]}
        )
        fig_countplot.update_layout(
            xaxis_title="Categorías", 
            yaxis_title="Cantidad de Pacientes",
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        # 2. GRÁFICA BOXPLOT (Tu análisis de rangos y distribución)
        fig_boxplot = px.box(
            df,
            x='risk_category',
            y='cholesterol_mg_dl',
            title="Distribución de Colesterol por Categoría de Riesgo",
            category_orders={'risk_category': ['Low', 'Medium', 'High']},
            template="plotly_white"
        )
        fig_boxplot.update_traces(marker_color='#148F77')
        fig_boxplot.update_layout(
            xaxis_title="Categorías", 
            yaxis_title="Colesterol (mg/dL)",
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        #3 GRAFICO HISTOGRAMA 
        fig_histo = px.histogram(data_frame=df,
                                 x='risk_category',
                                 )
        
        #Grafico de correlacion
        corr = df.corr(numeric_only=True)

        fig_heatmap = px.imshow(
            corr,
            text_auto=".2f",           
            color_continuous_scale="Blues",
            zmin=-1,
            zmax=1,
            aspect="auto"
        )

        fig_heatmap.update_layout(
            title="Matriz de Correlación",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=11)
        )
            
        df_numerico = df.select_dtypes(include=['float64', 'int64'])
        
        # 2. Calculamos la matriz de correlación real
        matriz_corr = df_numerico.corr()
        
        # 3. Extraemos la relación con la variable objetivo y quitamos su propia fila
        serie_corr = matriz_corr['heart_disease_risk_score'].abs().drop('heart_disease_risk_score')
        
        # 4. Creamos el DataFrame para tus barras
        df_altas_corr = serie_corr.reset_index()
        df_altas_corr.columns = ['Variable', 'Correlacion']
        df_altas_corr = df_altas_corr.sort_values('Correlacion', ascending=True) 

        # diagrama de barra de correlacion 
        fig_barcorr = px.bar(
            df_altas_corr.tail(15), 
            x="Correlacion", 
            y="Variable", 
            orientation='h',
            title="Top Factores de Mayor Incidencia",
            template="plotly_white"
        )

        fig_barcorr.update_traces(
            marker_color="#16ace3", 
            texttemplate='%{x:.2f}', 
            textposition='outside'
        )
        fig_barcorr.update_layout(
            xaxis_title="Coeficiente de Correlación ",
            yaxis_title="",
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin={'l': 150, 'r': 50, 't': 50, 'b': 40}
        )
        return html.Div(style={'padding': '20px'}, children=[
            html.H2('Análisis de Riesgo Cardiovascular', style={'color': ESTILOS['azul'], 'textAlign': 'center', 'marginBottom': '20px'}),
            
            
            html.Div(style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'}, children=[
                
                
                html.Div(style={'flex': 1, 'minWidth': '400px', 'backgroundColor': ESTILOS['panel'], 'padding': '20px', 'borderRadius': '8px', 'border': ESTILOS['borde']}, children=[
                    dcc.Graph(figure=fig_countplot)
                ]),
                
                
               html.Div(style={'flex': 1, 'minWidth': '400px', 'backgroundColor': ESTILOS['panel'], 'padding': '20px', 'borderRadius': '8px', 'border': ESTILOS['borde']}, children=[
                    html.Label('Variable Métrica para Boxplot:', style={'fontWeight': 'bold', 'color': ESTILOS['azul'], 'marginBottom': '8px', 'display': 'block'}),
                    
                    dcc.Dropdown(
                        id='dropdown-variable-boxplot',
                        options=[
                            {'label': 'Colesterol', 'value': 'cholesterol_mg_dl'},
                            {'label': 'Horas de Sueño', 'value': 'sleep_hours'},
                            {'label': 'Pasos Diarios', 'value': 'daily_steps'},
                            {'label': 'Nivel de Estrés', 'value': 'stress_level'},
                            {'label': 'Índice de Masa Corporal (BMI)', 'value': 'bmi'},
                            {'label': 'Edad', 'value': 'age'}
                        ],
                        value='cholesterol_mg_dl',
                        clearable=False
                    ),
                    html.Hr(style={'borderTop': ESTILOS['borde'], 'margin': '15px 0'}),
                
                    html.Div(id='contenedor-grafico-boxplot')
                ])
            ]),
            
            html.Div(style={'backgroundColor': ESTILOS['panel'], 'padding': '25px', 'borderRadius': '8px', 'border': ESTILOS['borde']}, children=[
                html.H4('Análisis de Relación Numérica (Matriz de Correlación)', style={'color': ESTILOS['azul'], 'margin': '0 0 10px 0'}),
                html.P('Estudio de coeficientes estadísticos para identificar variables de alta incidencia cardíaca.', style={'fontSize': '0.9em', 'color': '#566573', 'marginBottom': '15px'}),
                
                
                 html.Div(style={'flex': 1, 'minWidth': '400px', 'backgroundColor': ESTILOS['panel'], 'padding': '20px', 'borderRadius': '8px', 'border': ESTILOS['borde']}, children=[
                    dcc.Graph(figure=fig_heatmap)
                ])
            ]),
            
             html.Div(style={'backgroundColor': ESTILOS['panel'], 'padding': '25px', 'borderRadius': '8px', 'border': ESTILOS['borde']}, children=[
                html.H4('', style={'color': ESTILOS['azul'], 'margin': '0 0 10px 0'}),
                html.P('', style={'fontSize': '0.9em', 'color': '#566573', 'marginBottom': '15px'}),
                
                
                 html.Div(style={'flex': 1, 'minWidth': '400px', 'backgroundColor': ESTILOS['panel'], 'padding': '20px', 'borderRadius': '8px', 'border': ESTILOS['borde']}, children=[
                    dcc.Graph(figure=fig_barcorr)
                ])
            ]),
        ])
        
    elif tab == 'tab-prediccion':
        return html.Div(style={'padding': '30px'}, children=[
            html.H2('Predicción de Riesgo Cardiovascular en Tiempo Real', style={'color': '#00fff0', 'textAlign': 'center', 'marginBottom': '30px'}),
            
            # Contenedor principal de dos columnas
            html.Div(style={'display': 'flex', 'gap': '30px', 'flexWrap': 'wrap'}, children=[
                
                # COLUMNA IZQUIERDA: Formulario de Entrada (Inputs)
                html.Div(style={'flex': '1.5', 'minWidth': '400px', 'backgroundColor': '#1f1f1f', 'padding': '25px', 'borderRadius': '10px', 'border': ESTILOS['borde']}, children=[
                    html.H4("Datos de la Instancia del Paciente", style={'color': '#00fff0', 'marginBottom': '20px'}),
                    
                    # Fila 1: Datos Básicos
                    html.Div(style={'display': 'flex', 'gap': '15px', 'marginBottom': '15px'}, children=[
                        html.Div(style={'flex': 1}, children=[html.Label("Edad del Paciente:", style={'color': 'white'}), dcc.Input(id='input-edad', type='number', value=35, style={'width': '100%', 'padding': '8px', 'backgroundColor': '#2d2d2d', 'color': 'white', 'border': '1px solid #555'})]),
                        html.Div(style={'flex': 1}, children=[html.Label("Índice de Masa Corporal (BMI):", style={'color': 'white'}), dcc.Input(id='input-bmi', type='number', value=25.0, style={'width': '100%', 'padding': '8px', 'backgroundColor': '#2d2d2d', 'color': 'white', 'border': '1px solid #555'})]),
                    ]),

                    # Fila 2: Presión Arterial
                    html.Div(style={'display': 'flex', 'gap': '15px', 'marginBottom': '15px'}, children=[
                        html.Div(style={'flex': 1}, children=[html.Label("P. Arterial Sistólica (mmHg):", style={'color': 'white'}), dcc.Input(id='input-systolic', type='number', value=120, style={'width': '100%', 'padding': '8px', 'backgroundColor': '#2d2d2d', 'color': 'white', 'border': '1px solid #555'})]),
                        html.Div(style={'flex': 1}, children=[html.Label("P. Arterial Diastólica (mmHg):", style={'color': 'white'}), dcc.Input(id='input-diastolic', type='number', value=80, style={'width': '100%', 'padding': '8px', 'backgroundColor': '#2d2d2d', 'color': 'white', 'border': '1px solid #555'})]),
                    ]),

                    # Fila 3: Laboratorio y Ritmo
                    html.Div(style={'display': 'flex', 'gap': '15px', 'marginBottom': '15px'}, children=[
                        html.Div(style={'flex': 1}, children=[html.Label("Colesterol Total (mg/dL):", style={'color': 'white'}), dcc.Input(id='input-colesterol', type='number', value=190, style={'width': '100%', 'padding': '8px', 'backgroundColor': '#2d2d2d', 'color': 'white', 'border': '1px solid #555'})]),
                        html.Div(style={'flex': 1}, children=[html.Label("Ritmo Cardíaco en Reposo:", style={'color': 'white'}), dcc.Input(id='input-resting-heart', type='number', value=72, style={'width': '100%', 'padding': '8px', 'backgroundColor': '#2d2d2d', 'color': 'white', 'border': '1px solid #555'})]),
                    ]),

                    # Fila 4: Actividad Diaria
                    html.Div(style={'display': 'flex', 'gap': '15px', 'marginBottom': '15px'}, children=[
                        html.Div(style={'flex': 1}, children=[html.Label("Pasos Diarios:", style={'color': 'white'}), dcc.Input(id='input-steps', type='number', value=7000, style={'width': '100%', 'padding': '8px', 'backgroundColor': '#2d2d2d', 'color': 'white', 'border': '1px solid #555'})]),
                        html.Div(style={'flex': 1}, children=[html.Label("Horas de Ejercicio por Semana:", style={'color': 'white'}), dcc.Input(id='input-physical-hours', type='number', value=3, style={'width': '100%', 'padding': '8px', 'backgroundColor': '#2d2d2d', 'color': 'white', 'border': '1px solid #555'})]),
                    ]),

                    # Fila 5: Hábitos de Vida
                    html.Div(style={'display': 'flex', 'gap': '15px', 'marginBottom': '15px'}, children=[
                        html.Div(style={'flex': 1}, children=[html.Label("Horas de Sueño Diarias:", style={'color': 'white'}), dcc.Input(id='input-sueno', type='number', value=7, style={'width': '100%', 'padding': '8px', 'backgroundColor': '#2d2d2d', 'color': 'white', 'border': '1px solid #555'})]),
                        html.Div(style={'flex': 1}, children=[html.Label("Unidades de Alcohol por Semana:", style={'color': 'white'}), dcc.Input(id='input-alcohol', type='number', value=1, style={'width': '100%', 'padding': '8px', 'backgroundColor': '#2d2d2d', 'color': 'white', 'border': '1px solid #555'})]),
                    ]),

                    # Fila 6: Calificaciones y Estrés
                    html.Div(style={'display': 'flex', 'gap': '15px', 'marginBottom': '15px'}, children=[
                        html.Div(style={'flex': 1}, children=[html.Label("Nivel de Estrés (1-10):", style={'color': 'white'}), dcc.Input(id='input-stress', type='number', value=5, style={'width': '100%', 'padding': '8px', 'backgroundColor': '#2d2d2d', 'color': 'white', 'border': '1px solid #555'})]),
                        html.Div(style={'flex': 1}, children=[html.Label("Calidad de la Dieta (1-10):", style={'color': 'white'}), dcc.Input(id='input-diet-score', type='number', value=6, style={'width': '100%', 'padding': '8px', 'backgroundColor': '#2d2d2d', 'color': 'white', 'border': '1px solid #555'})]),
                    ]),

                    # Fila 7: Estados Binarios
                    html.Div(style={'display': 'flex', 'gap': '15px', 'marginBottom': '25px'}, children=[
                        html.Div(style={'flex': 1}, children=[
                            html.Label("¿Estado de Fumador?", style={'color': 'white'}),
                            dcc.Dropdown(id='dropdown-smoking', options=[{'label': 'No Fumador', 'value': 0}, {'label': 'Fumador', 'value': 1}], value=0, style={'color': 'black'})
                        ]),
                        html.Div(style={'flex': 1}, children=[
                            html.Label("Historial Familiar Cardíaco:", style={'color': 'white'}),
                            dcc.Dropdown(id='dropdown-history', options=[{'label': 'Sin Antecedentes', 'value': 0}, {'label': 'Con Antecedentes', 'value': 1}], value=0, style={'color': 'black'})
                        ]),
                    ]),

                    html.Button('Propagar Inferencia SVM', id='btn-predecir', n_clicks=0, style={
                        'width': '100%', 'backgroundColor': '#00fff0', 'color': 'black', 'fontWeight': 'bold', 'padding': '12px', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'
                    })
                ]),
                
                # COLUMNA DERECHA: Contenedor del Resultado 
                html.Div(style={'flex': '1', 'minWidth': '300px', 'backgroundColor': '#1f1f1f', 'padding': '25px', 'borderRadius': '10px', 'border': ESTILOS['borde']}, children=[
                    html.H4("Resultado del Diagnóstico", style={'color': '#00fff0', 'marginBottom': '20px'}),
                    html.Div(id='contenedor-resultado-prediccion')  
                ])
            ])
        ])
        # muestra el mapa y barras originales
    elif tab == 'tab-conectividad':
        fig_map = px.choropleth_mapbox(gdf_merged, geojson=geojson_dict, locations=gdf_merged.index, color="Valor",
                                      mapbox_style="carto-darkmatter", center={"lat": 8.53, "lon": -80.78}, zoom=6.5,
                                      color_continuous_scale="YlOrRd", template="plotly_dark")
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        
        fig_bar = px.bar(df_inec.sort_values("Valor").tail(15), x="Nombre Distrito", y="Valor", orientation='h', template="plotly_dark", color_discrete_sequence=['#d9381e'])
        
        return html.Div(style={"padding": "30px"}, children=[
            html.Div(style={"display": "flex", "gap": "20px"}, children=[
                html.Div(style={"flex": "2", "backgroundColor": "#1f1f1f", "padding": "20px", "borderRadius": "10px"}, children=[
                    html.H3("Mapa de Acceso a Celular", style={"color": "#00fff0"}),
                    dcc.Graph(figure=fig_map)
                ]),
                
            ]),
            
            html.Div(style={"flex": "1", "backgroundColor": "#1f1f1f", "padding": "20px", "borderRadius": "10px"}, children=[
                    html.H3("Top 15 Distritos", style={"color": "#00fff0"}),
                    dcc.Graph(figure=fig_bar)
                ])
        ])

#  ACTUALIZAR EL BOXPLOT DINÁMICAMENTE
@app.callback(
    Output('contenedor-grafico-boxplot', 'children'),
    [Input('dropdown-variable-boxplot', 'value')]
)
def actualizar_boxplot(variable_seleccionada):
    # Diccionario para asignar títulos limpios en español en la gráfica
    titulos = {
        'cholesterol_mg_dl': "Distribución de Colesterol por Categoría de Riesgo",
        'sleep_hours': "Distribución de Horas de Sueño por Categoría de Riesgo",
        'daily_steps': "Distribución de Pasos Diarios por Categoría de Riesgo",
        'stress_level': "Distribución de Niveles de Estrés por Categoría de Riesgo",
        'bmi': "Distribución de Índice de Masa Corporal por Categoría de Riesgo",
        'age': "Distribución de Edad por Categoría de Riesgo"
    }

    # Creamos el gráfico apuntando el eje Y a la variable seleccionada del Dropdown
    fig_dinamica = px.box(
        df,
        x='risk_category',
        y=variable_seleccionada,
        title=titulos.get(variable_seleccionada, "Distribución de Variable"),
        category_orders={'risk_category': ['Low', 'Medium', 'High']},
        template="plotly_white"
    )
    
    # Aplicamos estilos consistentes para que haga juego con el panel
    fig_dinamica.update_traces(marker_color='#148F77')
    fig_dinamica.update_layout(
        xaxis_title="Categorías de Riesgo", 
        yaxis_title=variable_seleccionada.replace('_', ' ').upper(),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin={'l': 40, 'r': 40, 't': 40, 'b': 40}
    )
    
    # Retornamos el componente gráfico actualizado listo para el layout
    return dcc.Graph(figure=fig_dinamica)   
        
        


# CALLBACK DE CLASIFICACIÓN CON TU MODELO SVM CLASIFICADOR REAL
@app.callback(
    Output('contenedor-resultado-prediccion', 'children'),
    [Input('btn-predecir', 'n_clicks')],
    [
        dash.dependencies.State('input-edad', 'value'),
        dash.dependencies.State('input-bmi', 'value'),
        dash.dependencies.State('input-systolic', 'value'),
        dash.dependencies.State('input-diastolic', 'value'),
        dash.dependencies.State('input-colesterol', 'value'),
        dash.dependencies.State('input-resting-heart', 'value'),
        dash.dependencies.State('input-steps', 'value'),
        dash.dependencies.State('dropdown-smoking', 'value'),
        dash.dependencies.State('input-stress', 'value'),
        dash.dependencies.State('input-physical-hours', 'value'),
        dash.dependencies.State('input-sueno', 'value'),
        dash.dependencies.State('input-diet-score', 'value'),
        dash.dependencies.State('input-alcohol', 'value'),
        dash.dependencies.State('dropdown-history', 'value')
    ]
)
def clasificar_con_svm_real_completo(n_clicks, edad, bmi, systolic, diastolic, colesterol, resting_heart, steps, smoking, stress, physical_hours, sueno, diet_score, alcohol, history):
    # Verificación de clicks iniciales o campos vacíos
    lista_inputs = [edad, bmi, systolic, diastolic, colesterol, resting_heart, steps, smoking, stress, physical_hours, sueno, diet_score, alcohol, history]
    if n_clicks == 0 or any(v is None for v in lista_inputs):
        return html.Div(style={'textAlign': 'center'}, children=[
            html.H2('Esperando Instancia...', style={'color': '#566573'}),
            html.P('Modifique los parámetros del paciente y haga clic en Propagar Inferencia SVM.', style={'color': '#85929e'})
        ])
    
    # 1. el diccionario con los nombres exactos de características
    datos_paciente = {
        'age': edad,
        'bmi': bmi,
        'systolic_bp': systolic,
        'diastolic_bp': diastolic,
        'cholesterol_mg_dl': colesterol,
        'resting_heart_rate': resting_heart,
        'daily_steps': steps,
        'smoking_status': smoking,
        'stress_level': stress,
        'physical_activity_hours_per_week': physical_hours,
        'sleep_hours': sueno,
        'diet_quality_score': diet_score,
        'alcohol_units_per_week': alcohol,
        'family_history_heart_disease': history
    }
    
    # DataFrame inicial
    instancia_cruda = pd.DataFrame([datos_paciente])
    
    # 2. alineación de columnas idéntica al fit del StandardScaler
    orden_estricto_fit = list(escalador_real.feature_names_in_)
    instancia_cruda = instancia_cruda[orden_estricto_fit]
    
    # 3. Escalado e inferencia directa con tu modelo cargado (.pkl)
    instancia_escalada = escalador_real.transform(instancia_cruda)
    
    clase_predicha_cruda = modelo_real_svm.predict(instancia_escalada)[0]
    probabilidades = modelo_real_svm.predict_proba(instancia_escalada)[0]
    
    # Mapeo estructurado de las probabilidades asociadas a las clases del SVM
    clases_modelo = list(modelo_real_svm.classes_)
    dict_probabilidades = dict(zip(clases_modelo, probabilidades))
    
    prob_low = dict_probabilidades.get('Low', 0.0) * 100
    prob_med = dict_probabilidades.get('Medium', 0.0) * 100
    prob_high = dict_probabilidades.get('High', 0.0) * 100

    colores_dict = {"High": "#C0392B", "Medium": "#D35400", "Low": "#148F77"}
    color_clase = colores_dict.get(clase_predicha_cruda, "#566573")

    return html.Div([
        html.Div(style={'textAlign': 'center', 'marginBottom': '25px'}, children=[
            html.P("PREDICCIÓN (SVM):", style={'color': '#85929e', 'fontSize': '0.9em', 'letterSpacing': '1px'}),
            html.H2(f"{clase_predicha_cruda.upper()}", style={'color': color_clase, 'fontWeight': 'bold', 'fontSize': '2.2em', 'margin': '5px 0'}),
            html.Span("Precisión del Modelo en Testeo: 91.82%", style={'color': '#00fff0', 'fontSize': '0.85em', 'fontWeight': 'bold'})
        ]),
        
        html.Hr(style={'borderColor': '#2b2b3d', 'marginBottom': '20px'}),
        
        html.H5("Distribución de Probabilidades Reales (predict_proba):", style={'color': 'white', 'marginBottom': '15px'}),
        
        # Barra Low
        html.Div(style={'marginBottom': '12px'}, children=[
            html.Div(style={'display': 'flex', 'color': '#85929e', 'fontSize': '0.9em'}, children=[
                html.Span("Probabilidad de Riesgo Bajo (Low):"),
                html.Span(f"{prob_low:.1f}%", style={'fontWeight': 'bold', 'color': '#148F77', 'marginLeft': 'auto'})
            ]),
            html.Div(style={'backgroundColor': '#2b2b3d', 'height': '8px', 'borderRadius': '4px', 'marginTop': '4px'}, children=[
                html.Div(style={'backgroundColor': '#148F77', 'width': f'{prob_low}%', 'height': '100%', 'borderRadius': '4px'})
            ])
        ]),

        # Barra Medium
        html.Div(style={'marginBottom': '12px'}, children=[
            html.Div(style={'display': 'flex', 'color': '#85929e', 'fontSize': '0.9em'}, children=[
                html.Span("Probabilidad de Riesgo Medio (Medium):"),
                html.Span(f"{prob_med:.1f}%", style={'fontWeight': 'bold', 'color': '#D35400', 'marginLeft': 'auto'})
            ]),
            html.Div(style={'backgroundColor': '#2b2b3d', 'height': '8px', 'borderRadius': '4px', 'marginTop': '4px'}, children=[
                html.Div(style={'backgroundColor': '#D35400', 'width': f'{prob_med}%', 'height': '100%', 'borderRadius': '4px'})
            ])
        ]),

        # Barra High
        html.Div(style={'marginBottom': '5px'}, children=[
            html.Div(style={'display': 'flex', 'color': '#85929e', 'fontSize': '0.9em'}, children=[
                html.Span("Probabilidad de Riesgo Alto (High):"),
                html.Span(f"{prob_high:.1f}%", style={'fontWeight': 'bold', 'color': '#C0392B', 'marginLeft': 'auto'})
            ]),
            html.Div(style={'backgroundColor': '#2b2b3d', 'height': '8px', 'borderRadius': '4px', 'marginTop': '4px'}, children=[
                html.Div(style={'backgroundColor': '#C0392B', 'width': f'{prob_high}%', 'height': '100%', 'borderRadius': '4px'})
            ])
        ]),
        
        html.P(" Inferencia computada exitosamente utilizando el vector de 14 dimensiones reales.", 
               style={'color': '#566573', 'fontSize': '0.8em', 'marginTop': '25px', 'textAlign': 'center'})
    ])
        

if __name__ == "__main__":
    app.run(debug=True)