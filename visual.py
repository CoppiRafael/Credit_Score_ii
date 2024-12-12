

import warnings
import math

import pandas                  as pd
import numpy                   as np
import matplotlib.pyplot       as plt
import seaborn                 as sns
import plotly.express          as px
import statsmodels.api         as sm
import statsmodels.formula.api as smf
import xgboost                 as xgb
import scipy.stats             as stats
import streamlit               as st
import streamlit.components.v1 as components

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn                              import metrics
from scipy.stats                          import ks_2samp,t,entropy,chi2_contingency
from pycaret                              import *
from sklearn.model_selection              import train_test_split
from sklearn.model_selection              import GridSearchCV,RandomizedSearchCV
from sklearn.impute                       import SimpleImputer
from sklearn.metrics                      import accuracy_score, roc_auc_score, classification_report,mean_squared_error
from sklearn.compose                      import ColumnTransformer
from sklearn.pipeline                     import Pipeline
from sklearn.preprocessing                import OneHotEncoder, StandardScaler
from sklearn.preprocessing                import LabelEncoder
from sklearn.decomposition                import PCA
from sklearn.linear_model                 import LogisticRegression
from scipy.stats                          import uniform,lognorm
from pycaret.classification               import *
st.set_page_config(
        layout='wide',
        initial_sidebar_state='expanded'
    )

custom_params = {"axes.spines.right":False,"axes.spines.top":False}
sns.set_theme(style='ticks',rc=custom_params)
color_palette = 'vlag'
sns.set_palette(sns.color_palette(color_palette))



def main():
    
    components.html("""
        <script src="https://cdnjs.cloudflare.com/ajax/libs/bodymovin/5.7.6/lottie.min.js"></script>
        <div id="lottie" style="width: 200px; height: 200px; margin: auto;"></div>
        <script>
            var animation = bodymovin.loadAnimation({
                container: document.getElementById('lottie'),
                renderer: 'svg',
                loop: true,
                autoplay: true,
                path: 'https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json' // Link para animação Lottie
            });
        </script>
    """, height=200)
    st.markdown(
        """
        <div style="text-align: center;">
            <h1 style="color: #2a3e48;">CREDIT SCORING</h1>
            <h3 style="color: #2a3e48;">Projeto Final EBAC</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    trade_history_file = st.file_uploader("Upload Trade History File", type=['csv','ftr'])
    st.divider()


    df = pd.read_feather(trade_history_file) if trade_history_file is not None else None

    if df is not None:
        df = df.sort_values(by='data_ref')
        data_corte = df.data_ref.max() - pd.DateOffset(months=3)
        df_out_of_time = df[df['data_ref'] > data_corte]
        df_treino = df[df['data_ref'] <= data_corte]
        df_ = df_treino.copy()
        tab1,tab2, tab3, tab4, tab5, tab6 = st.tabs(["Visualização dos Dados","Análise Descritiva Categorica", "Análise Descritiva Numérica","Weight of Evidence (WoE)",'Information Value (IV)','Modelo Machine Learning'])

        with tab1:
            st.subheader("Visualização e Métricas do DataFrame")
    
            # Adicionando filtros para as colunas do DataFrame
            st.markdown("### Filtros")
            col1_filter, col2_filter, col3_filter = st.columns(3)
            
            with col1_filter:
                selected_columns = st.multiselect(
                    "Selecione as colunas para exibir:",
                    df.columns,
                    default=df.columns[:3]  # Pré-selecionando algumas colunas
                )
            
            with col2_filter:
                num_rows = st.slider(
                    "Número de linhas para visualizar:",
                    min_value=5, max_value=len(df), value=8
                )
            
            with col3_filter:
                sort_column = st.selectbox(
                    "Ordenar por coluna:",
                    options=["Nenhuma"] + list(df.columns),
                    index=0
                )
            st.divider()
            # Aplicando os filtros
            filtered_df = df[selected_columns]
            if sort_column != "Nenhuma":
                filtered_df = filtered_df.sort_values(by=sort_column)
            
            # Dividindo em colunas para visualização
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Prévia do DataFrame")
                st.write(filtered_df.head(num_rows))
            
            with col2:
                st.markdown("### Métricas Descritivas")
                st.write(df[selected_columns].describe())
            st.divider()
            # Métricas adicionais do DataFrame
            st.markdown("### Métricas do DataFrame")
            col3, col4 = st.columns(2)
            
            with col3:
                st.metric("Total de Linhas", df.shape[0])
                st.metric("Total de Colunas", df.shape[1])
            
            with col4:
                st.metric("Valores Nulos", df.isnull().sum().sum())
                st.metric("Duplicatas", df.duplicated().sum())

            if df.duplicated().sum() > 0:
                st.markdown('---')
                st.markdown('Sobre os duplicados:')
                st.write(df.duplicated())
        with tab2:
            st.subheader("Distribuição das Variáveis Categóricas com base em 'mau'")
            cat = df_.select_dtypes(['object'])
            cat_var = cat.columns.tolist()
            
            num_vars = len(cat_var)
            num_cols = 2 if num_vars <= 6 else 3
            num_rows = math.ceil(num_vars / num_cols)

            current_var_index = 0  

            for row in range(num_rows):
                cols = st.columns(num_cols)  
                for col in cols:
                    if current_var_index < num_vars:
                        var = cat_var[current_var_index]

                        fig, ax = plt.subplots(figsize=(5, 5))  
                        sns.countplot(data=df_, x=var, hue='mau', ax=ax)
                        for p in ax.patches:
                            ax.annotate(f'{int(p.get_height())}',
                                        (p.get_x() + p.get_width() / 2., p.get_height()),
                                        ha='center', va='baseline',
                                        fontsize=10, color='black', xytext=(0, 5),
                                        textcoords='offset points')
                        ax.set_title(f'Countplot de {var}', fontsize=12)
                        ax.set_xlabel(var, fontsize=10)
                        ax.set_ylabel('Contagem', fontsize=10)
                        plt.tight_layout()

                        # Mostrar o gráfico na coluna atual
                        col.pyplot(fig)
                        current_var_index += 1
            st.info("""
                **Insights gerais sobre os gráficos:**
                - **Comparação entre categorias e a variável `mau`**: Cada gráfico mostra como as categorias das variáveis qualitativas se distribuem entre as classes de `mau`. Isso permite avaliar se algumas categorias estão mais associadas a uma classe específica.
                - **Tendências nas variáveis**: É possível observar se algumas categorias de variáveis se tornam mais prevalentes em determinados períodos ou em uma das classes de `mau`, sugerindo tendências ou padrões.
                - **Identificação de desequilíbrios**: Alguns gráficos podem revelar desequilíbrios entre as classes de `mau` para uma categoria, o que pode ser relevante para ajustes no modelo ou técnicas de balanceamento.
                - **Distribuição visual clara**: A diferenciação das barras por cores facilita a visualização rápida de como as categorias se distribuem ao longo das classes de `mau`, destacando possíveis correlações entre variáveis qualitativas e o desfecho.
            """)
            st.divider()
            st.subheader('Análise temporal de variáveis Qualitativas')
            st.info("""
                Os gráficos a seguir mostram a contagem de observações para cada categoria das variáveis qualitativas ao longo do tempo. 
                A variável `new_data` é representada no eixo X, agrupada por ano e mês, e o eixo Y mostra a contagem das ocorrências de cada categoria de cada variável qualitativa.
                Esses gráficos ajudam a identificar tendências e padrões ao longo do tempo, como o aumento ou diminuição de categorias específicas, 
                o comportamento de variáveis ao longo de diferentes períodos, e possíveis variações relacionadas a eventos específicos.
                """)
            df['data_ref'] = pd.to_datetime(df['data_ref'])
            df['new_data'] = df['data_ref'].apply(lambda x: x.strftime('%Y-%m'))
            var_qualitative = df.drop(['mau','new_data','data_ref'],axis=1).select_dtypes(exclude='number').columns.tolist()

            for var in var_qualitative:
                plt.figure(figsize=(12,6))  # Tamanho do gráfico
                df_grouped = (
                    df[['new_data', var, 'mau']]
                    .groupby(['new_data', var])
                    .count()
                    .reset_index()
                    .rename(columns={'mau': 'count'})
                    .sort_values('new_data')
                )
                
                # Gerar gráfico de linha
                sns.lineplot(data=df_grouped, x='new_data', y='count', hue=var, marker='o')
                plt.title(f'Média de {var} ao longo dos anos')
                plt.xticks(rotation=90)
                plt.xlabel('Ano')
                plt.ylabel('Contagem')
                plt.tight_layout()
                st.pyplot(plt)  # Exibir gráfico no Streamlit

            st.info("""
                **Insights gerais sobre os gráficos:**
                - **Padrões sazonais**: Os gráficos mostram como as categorias das variáveis qualitativas mudam ao longo do tempo. É possível identificar períodos de maior ou menor prevalência de determinadas categorias.
                - **Tendências gerais**: O comportamento das variáveis ao longo dos anos pode revelar tendências de crescimento ou diminuição de certas categorias.
                - **Anomalias**: Alguns gráficos podem revelar picos ou quedas inesperadas, que podem ser indicativos de eventos ou mudanças externas que impactaram as variáveis analisadas.
                - **Comparações entre categorias**: Ao observar a variação de diferentes categorias de uma variável ao longo do tempo, podemos comparar qual categoria tem mais incidência em determinados períodos e como as outras categorias se comportam.
            """)   
            st.divider()
            st.info("""
        **Análise de Entropia e Gini:**
        Esta análise avalia a distribuição das categorias dentro de cada variável categórica usando métricas como Entropia e Gini.
        - **Entropia**: Mede a aleatoriedade ou desordem da distribuição das categorias.
        - **Índice de Gini**: Avalia a pureza das categorias. Quanto menor o valor, mais homogênea é a variável.
        - Essa análise é útil para identificar quais variáveis podem ser mais relevantes para um modelo preditivo.
    """)

            gini_results = []
            entropy_results = []

            for var in cat_var:
                counts = df_[var].value_counts(normalize=True)
                gini = 1 - sum(counts**2)
                gini_results.append((var, gini))

                ent = entropy(counts)
                entropy_results.append((var, ent))

            gini_df = pd.DataFrame(gini_results, columns=['Variável', 'Índice de Gini']).sort_values('Índice de Gini', ascending=True)
            entropy_df = pd.DataFrame(entropy_results, columns=['Variável', 'Entropia']).sort_values('Entropia', ascending=False)
            
            col1,col2 = st.columns(2)
            with col1:

                st.write("### Top Variáveis por Gini (mais homogêneas):")
                st.dataframe(gini_df.head(5))
            with col2: 
                st.write("### Top Variáveis por Entropia (mais diversas):")
                st.dataframe(entropy_df.head(5))
            st.divider()
            col1,col2 = st.columns(2)
            with col1:
            # Gráficos adicionais para Gini e Entropia
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=gini_df, x='Índice de Gini', y='Variável', ax=ax, palette='coolwarm')
                ax.set_title("Índice de Gini das Variáveis Categóricas")
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=entropy_df, x='Entropia', y='Variável', ax=ax, palette='coolwarm')
                ax.set_title("Entropia das Variáveis Categóricas")
                st.pyplot(fig)

            # Análise 2: Detecção de Interações entre Variáveis Categóricas
            st.info("""
                **Interação entre Variáveis Categóricas:**
                Identificamos interações entre variáveis categóricas através de tabelas de contingência e testes estatísticos como o teste Qui-Quadrado.
                - Essas interações ajudam a identificar dependências entre variáveis, o que pode ser relevante na construção de modelos.
                - Por exemplo, variáveis fortemente dependentes podem ser redundantes ou indicar um fator comum subjacente.
            """)
            col1,col2 = st.columns(2)
            chi2_results = []

            for i in range(len(cat_var)):
                for j in range(i + 1, len(cat_var)):
                    var1 = cat_var[i]
                    var2 = cat_var[j]

                    contingency_table = pd.crosstab(df_[var1], df_[var2])
                    chi2, p, dof, expected = chi2_contingency(contingency_table)

                    chi2_results.append((var1, var2, chi2, p))

            chi2_df = pd.DataFrame(chi2_results, columns=['Variável 1', 'Variável 2', 'Qui-Quadrado', 'P-Valor']).sort_values('P-Valor')
            with col1: 
                st.write("### Interações Significativas (P-Valor < 0.05):")
                st.dataframe(chi2_df[chi2_df['P-Valor'] < 0.05])

            # Gráfico de correlações categóricas
            #st.info("""
            #    O gráfico abaixo destaca as interações categóricas mais relevantes com base no valor do teste Qui-Quadrado.
            #""")
            with col2:
                fig, ax = plt.subplots(figsize=(12, 8))
                top_interactions = chi2_df[chi2_df['P-Valor'] < 0.05].head(10)
                sns.barplot(data=top_interactions, x='Qui-Quadrado', y='Variável 1', hue='Variável 2', dodge=False, ax=ax)
                ax.set_title("Interações entre Variáveis Categóricas (Qui-Quadrado)")
                st.pyplot(fig)

            # Análise 3: Teste de Independência com a Variável `mau`
            #st.info("""
            #    **Independência entre Categorias e 'mau':**
            #    Realizamos um teste estatístico para avaliar se as categorias de cada variável categórica são independentes da variável-alvo `mau`.
            #    - Teste utilizado: Qui-Quadrado
            #    - O objetivo é identificar as variáveis com maior associação à variável-alvo.
            #""")
            col1,col2 = st.columns(2)
            target_chi2_results = []
            st.divider()
            for var in cat_var:
                contingency_table = pd.crosstab(df_[var], df_['mau'])
                chi2, p, dof, expected = chi2_contingency(contingency_table)

                target_chi2_results.append((var, chi2, p))
            with col1:

                target_chi2_df = pd.DataFrame(target_chi2_results, columns=['Variável', 'Qui-Quadrado', 'P-Valor']).sort_values('P-Valor')
                st.write("### Variáveis Mais Associadas à `mau` (P-Valor < 0.05):")
                st.dataframe(target_chi2_df[target_chi2_df['P-Valor'] < 0.05])
            with col2:
            # Gráfico de associação com 'mau'
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=target_chi2_df[target_chi2_df['P-Valor'] < 0.05], x='Qui-Quadrado', y='Variável', ax=ax, palette='coolwarm')
                ax.set_title("Associação das Variáveis Categóricas com 'mau'")
                st.pyplot(fig)
        with tab3:
            num_cols = df_.select_dtypes(include='number').columns.drop('index')
            num_plots = len(num_cols)
            st.subheader("Distribuição das Variáveis Numéricas em Relação a 'mau'")
            fig, axes = plt.subplots(nrows=(num_plots // 3) + 1, ncols=3, figsize=(20, 13))
            axes = axes.flatten()

            for i, col in enumerate(num_cols):
                sns.histplot(data=df_, x=col, hue='mau', bins=30, ax=axes[i],alpha=1,multiple="stack")
                axes[i].set_title(f'Histograma de {col}')
                axes[i].grid(False)

            plt.tight_layout()
            st.pyplot(plt)
            st.info("""
            **Insights Possíveis:**
            1. **Sobreposição entre as Classes de `mau`**:
            - Identificar variáveis com maior ou menor capacidade discriminativa.
            2. **Assimetria e Outliers**:
            - Compreender a necessidade de transformações nos dados.
            3. **Distribuição Geral**:
            - Avaliar concentrações ou anomalias nos dados.
            4. **Comparação de Proporções**:
            - Como as proporções de `mau` variam ao longo dos valores das variáveis.
            5. **Relevância para Modelagem**:
            - Identificar variáveis promissoras para análises preditivas.
            """)
            st.subheader("Boxplots das Variáveis Numéricas em Relação a 'mau'")
            st.markdown("""
            Os boxplots a seguir mostram a distribuição das variáveis numéricas, segmentadas pela variável `mau`. 
            Esses gráficos são úteis para analisar diferenças entre as classes, outliers, e a dispersão dos dados.
            """)
            fig, axes = plt.subplots(nrows=(num_plots // 3) + 1, ncols=3, figsize=(20, 13))
            axes = axes.flatten()

            for i, col in enumerate(num_cols):
                sns.boxplot(data=df_, x=col,hue='mau', linewidth=.7,flierprops={"marker": "x"}, ax=axes[i])
                axes[i].set_title(f'Boxplot de {col}')
                axes[i].grid(False)

            plt.tight_layout()
            st.pyplot(plt)
            st.info("""
            **Insights Possíveis dos Boxplots:**
            1. Diferenças estatísticas entre classes, como medianas distintas.
            2. Identificação visual de outliers.
            3. Avaliação da dispersão e simetria dos dados.
            """)
            st.subheader("Análise de Relações com Pairplot")
            st.markdown("""
            O pairplot fornece uma visão combinada das distribuições univariadas e das relações entre pares de variáveis numéricas, permitindo 
            identificar correlações, clusters e padrões relevantes no dataset.
            """)
            fig2 = plt.figure()
            sns.pairplot(data=df_.drop(columns=['index']).select_dtypes(include='number'))
            st.pyplot(plt)
            st.info("""
            **Insights Possíveis do Pairplot:**
            1. Identificar correlações ou colinearidades entre variáveis.
            2. Avaliar dispersões gerais e clusters distintos.
            3. Identificar padrões visuais que podem diferenciar classes da variável `mau`.
            """)
        _ = df_.isna().sum().to_frame()
        df_.loc[:,'renda'] = np.log(df_['renda'])
        df_.rename(columns={'renda':'log_renda'},inplace=True)
        df_.loc[:,'tempo_emprego'] = np.log(df_['tempo_emprego'])
        df_.rename(columns={'tempo_emprego':'log_tempo_emprego'},inplace=True)
        df_fill = df_.copy()
        df_fill['sexo'] = df_['sexo'].apply(lambda x: 1 if x== 'F' else 0)
        df_fill['posse_de_veiculo'] = df_fill['posse_de_veiculo'].apply(lambda x: 1 if x== 'S' else 0)
        df_fill['posse_de_imovel'] = df_fill['posse_de_imovel'].apply(lambda x: 1 if x== 'S' else 0)
        df_fill = df_fill.drop(columns=['data_ref','index'])
        df_fill = pd.get_dummies(df_fill)
        df_fill.dropna(inplace=True)
        known_data = df_fill[df_fill['log_tempo_emprego'].notnull()]
        unknown_data = df_fill[df_fill['log_tempo_emprego'].isnull()]
        X_known = known_data.drop(columns=['log_tempo_emprego'])
        y_known = known_data['log_tempo_emprego']
        model = xgb.XGBRegressor(n_estimator=70,max_depth=7,learning_rate=0.1)
        model.fit(X_known, y_known)
        X_unknown = unknown_data.drop(columns=['log_tempo_emprego'])
        predicted_tempo_emprego = model.predict(X_unknown)
        df_fill.loc[df_fill['log_tempo_emprego'].isnull(), 'log_tempo_emprego'] = predicted_tempo_emprego
        lista_index = df_.loc[df_['log_tempo_emprego'].isnull(),:].index.tolist()
        df_1 = df_.drop(columns=['log_tempo_emprego'])
        df_1['sexo'] = df_['sexo'].apply(lambda x: 1 if x== 'F' else 0)
        df_1['posse_de_veiculo'] = df_1['posse_de_veiculo'].apply(lambda x: 1 if x== 'S' else 0)
        df_1['posse_de_imovel'] = df_1['posse_de_imovel'].apply(lambda x: 1 if x== 'S' else 0)
        df_1 = df_1.drop(columns=['data_ref','index'])
        df_1 = pd.get_dummies(df_1)
        
        predicted = model.predict(df_1.iloc[lista_index,:])
        df_.loc[df_['log_tempo_emprego'].isna(),'log_tempo_emprego'] = predicted
        metadados = (df_
             .drop(columns=['data_ref','index'])
             .dtypes
             .to_frame()
             .rename(columns={0:'dtype'})
                    )
        metadados['nmissings'] = df_.isna().sum()
        metadados['unique_val'] = df_.nunique()
        def IV(variavel, resposta):
            tab = pd.crosstab(variavel, resposta, margins=True, margins_name='total')

            rótulo_evento = tab.columns[0]
            rótulo_nao_evento = tab.columns[1]

            tab['pct_evento'] = tab[rótulo_evento]/tab.loc['total',rótulo_evento]
            tab['ep'] = tab[rótulo_evento]/tab.loc['total',rótulo_evento]

            tab['pct_nao_evento'] = tab[rótulo_nao_evento]/tab.loc['total',rótulo_nao_evento]
            tab['woe'] = np.log(tab.pct_evento/tab.pct_nao_evento)
            tab['iv_parcial'] = (tab.pct_evento - tab.pct_nao_evento)*tab.woe
            return tab['iv_parcial'].sum()
        iv_sexo = IV(df_.sexo, df_.mau)
        metadados = (df_
             .drop(columns=['data_ref','index'])
             .dtypes
             .to_frame()
             .rename(columns={0:'dtype'})
            )
        metadados['nmissings'] = df_.isna().sum()
        metadados['unique_val'] = df_.nunique()
        metadados['papel'] = 'covariavel'
        metadados.loc['mau','papel'] = 'resposta'
        metadados.loc['bom','papel'] = 'resposta'
        for var in metadados[metadados.papel=='covariavel'].index:
            if  (metadados.loc[var, 'unique_val']>6):
                metadados.loc[var, 'IV'] = IV(pd.qcut(df_[var],5,duplicates='drop'), df_.mau)
            else:
                metadados.loc[var, 'IV'] = IV(df_[var], df_.mau)
        def biv_discreta(var, df):
            df['bom'] = 1-df.mau
            g = df.groupby(var)

            biv = pd.DataFrame({'qt_bom': g['bom'].sum(),
                                'qt_mau': g['mau'].sum(),
                                'mau':g['mau'].mean(),
                                var: g['mau'].mean().index,
                                'cont':g[var].count()})

            biv['ep'] = (biv.mau*(1-biv.mau)/biv.cont)**.5
            biv['mau_sup'] = biv.mau+t.ppf([0.975], biv.cont-1)*biv.ep
            biv['mau_inf'] = biv.mau+t.ppf([0.025], biv.cont-1)*biv.ep

            biv['logit'] = np.log(biv.mau/(1-biv.mau))
            biv['logit_sup'] = np.log(biv.mau_sup/(1-biv.mau_sup))
            biv['logit_inf'] = np.log(biv.mau_inf/(1-biv.mau_inf))

            tx_mau_geral = df.mau.mean()
            woe_geral = np.log(df.mau.mean() / (1 - df.mau.mean()))

            biv['woe'] = biv.logit - woe_geral
            biv['woe_sup'] = biv.logit_sup - woe_geral
            biv['woe_inf'] = biv.logit_inf - woe_geral

            fig, ax = plt.subplots(2,1, figsize=(8,6))
            ax[0].plot(biv[var], biv.woe, ':c', label='woe')
            ax[0].plot(biv[var], biv.woe_sup, 'o:m', label='limite superior')
            ax[0].plot(biv[var], biv.woe_inf, 'o:m', label='limite inferior')

            num_cat = biv.shape[0]
            ax[0].set_xlim([-.3, num_cat-.7])

            ax[0].set_ylabel("Weight of Evidence")
            ax[0].legend(bbox_to_anchor=(.83, 1.17), ncol=3)

            ax[0].set_xticks(list(range(num_cat)))
            ax[0].set_xticklabels(biv[var], rotation=15)

            ax[1] = biv.cont.plot.bar()
            st.pyplot(plt)
            return biv
        
        def biv_continua(var, ncat, df):
            df['bom'] = 1-df.mau
            cat_srs, bins = pd.qcut(df[var], ncat, retbins=True, precision=0, duplicates='drop')
            g = df.groupby(cat_srs)

            biv = pd.DataFrame({'qt_bom': g['bom'].sum(),
                                'qt_mau': g['mau'].sum(),
                                'mau':g['mau'].mean(),
                                var: g[var].mean(),
                                'cont':g[var].count()})

            biv['ep'] = (biv.mau*(1-biv.mau)/biv.cont)**.5
            biv['mau_sup'] = biv.mau+t.ppf([0.975], biv.cont-1)*biv.ep
            biv['mau_inf'] = biv.mau+t.ppf([0.025], biv.cont-1)*biv.ep

            biv['logit'] = np.log(biv.mau/(1-biv.mau))
            biv['logit_sup'] = np.log(biv.mau_sup/(1-biv.mau_sup))
            biv['logit_inf'] = np.log(biv.mau_inf/(1-biv.mau_inf))

            tx_mau_geral = df.mau.mean()
            woe_geral = np.log(df.mau.mean() / (1 - df.mau.mean()))

            biv['woe'] = biv.logit - woe_geral
            biv['woe_sup'] = biv.logit_sup - woe_geral
            biv['woe_inf'] = biv.logit_inf - woe_geral

            fig, ax = plt.subplots(2,1, figsize=(8,6))
            ax[0].plot(biv[var], biv.woe, ':c', label='woe')
            ax[0].plot(biv[var], biv.woe_sup, 'o:m', label='limite superior')
            ax[0].plot(biv[var], biv.woe_inf, 'o:m', label='limite inferior')

            num_cat = biv.shape[0]

            ax[0].set_ylabel("Weight of Evidence")
            ax[0].legend(bbox_to_anchor=(.83, 1.17), ncol=3)

            ax[1] = biv.cont.plot.bar()
            st.pyplot(plt)
            return None

        with tab4:
            st.markdown(
                """
                <div style="text-align: center;">
                    <h3>BIVARIADA DISCRETA</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                """
                A análise bivariada de variáveis discretas tem como objetivo explorar a relação entre uma variável explicativa categórica e a variável resposta, geralmente binária (bom/mau). Abaixo estão os gráficos para interpretar o **Weight of Evidence (WoE)**, bem como a distribuição das categorias:

                **Definições Importantes:**
                - **Weight of Evidence (WoE):** Uma métrica que indica a diferença logarítmica entre a taxa de eventos (mau) em uma categoria específica e a taxa geral de eventos. É útil para análise de variáveis preditoras em modelos de risco de crédito.
                - **Limites Superior e Inferior do WoE:** Representam a incerteza na estimativa de WoE baseada no tamanho da amostra.
                - **Distribuição das Categorias:** O gráfico de barras no eixo inferior mostra o número de observações em cada categoria.

                **Como interpretar:**
                - Valores de WoE positivos indicam categorias associadas a uma maior taxa de eventos (mau).
                - Valores negativos de WoE indicam categorias associadas a uma menor taxa de eventos (mau).
                - A estabilidade do modelo depende de categorias com volumes suficientes (ver gráfico de barras).
                """
            )

            for var in df_.select_dtypes(['object']):
                biv_discreta(var, df_)
            st.divider()
            st.markdown(
                """
                <div style="text-align: center;">
                    <h3>Remodelando algumas variáveis</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
            """
            Algumas variáveis foram remodeladas para reduzir a granularidade das categorias, agrupando categorias semelhantes. Isso melhora a estabilidade do modelo e a interpretabilidade dos resultados:

            - **Variável `tipo_renda`:** Categorias 'Bolsista' e 'Servidor público' foram agrupadas em 'Serv_P/Bols.'
            - **Variável `educacao`:** As categorias 'Médio' e 'Fundamental' foram agrupadas em 'Médio/Fund', enquanto categorias relacionadas ao ensino superior foram agrupadas em 'Sup_comp/incomp/Pos'.
            - **Variável `tipo_residencia`:** Categorias relacionadas à habitação governamental ou compartilhada foram agrupadas em 'Gov/c_pais/estud/comun'.
            - **Variável `estado_civil`:** Categorias relacionadas a uniões foram agrupadas em 'Casado/União', enquanto 'Separado' e 'Solteiro' foram agrupados em 'Separado/Solteiro'.

            Esses ajustes são fundamentais para análises mais robustas e redução de ruídos nos dados.
            """
            )


            df2 = df_.copy()
            df2.tipo_renda.replace({'Bolsista': 'Serv_P/Bols.', 'Servidor público': 'Serv_P/Bols.'}, inplace=True)
            biv_discreta('tipo_renda', df2)
            df2.educacao.replace({'Médio': 'Médio/Fund', 'Fundamental': 'Médio/Fund','Superior completo':'Sup_comp/incomp/Pos','Superior incompleto':'Sup_comp/incomp/Pos','Pós graduação':'Sup_comp/incomp/Pos'}, inplace=True)
            biv_discreta('educacao', df2)
            df2.tipo_residencia.replace({'Governamental':'Gov/c_pais/estud/comun','Com os pais':'Gov/c_pais/estud/comun','Estúdio':'Gov/c_pais/estud/comun','Comunitário':'Gov/c_pais/estud/comun'}, inplace=True)
            biv_discreta('tipo_residencia', df2)
            df2.estado_civil.replace({'Casado':'Casado/União','União':'Casado/União','Separado':'Separado/Solteiro','Solteiro':'Separado/Solteiro'}, inplace=True)
            biv_discreta('estado_civil', df2)

            st.markdown(
                """
                <div style="text-align: center;">
                    <h3>BIVARIADA CONTINUA</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                """
                A análise bivariada para variáveis contínuas envolve segmentar a variável em categorias utilizando **quartis** ou faixas definidas. Os gráficos gerados apresentam as seguintes informações:

                **Definições Importantes:**
                - **Weight of Evidence (WoE):** Calculado para cada faixa da variável contínua.
                - **Limites Superior e Inferior do WoE:** Determinados para cada faixa para indicar incertezas na estimativa.
                - **Distribuição da Frequência:** Gráficos de barras mostram o número de observações em cada faixa.

                **Como interpretar:**
                - Um padrão crescente ou decrescente no WoE ao longo das faixas pode indicar uma relação linear entre a variável contínua e a variável resposta.
                - A análise de frequência ajuda a verificar se há faixas com poucos dados, o que pode afetar a estabilidade do WoE.

                **Exemplos analisados:**
                - `qtd_filhos`: Segmentado em 4 faixas.
                - `idade`: Segmentado em 10 faixas.
                - `log_tempo_emprego` e `log_renda`: Segmentados em 10 faixas cada.

                Esses gráficos permitem verificar padrões de risco associados às variáveis contínuas, suportando decisões de modelagem.
                """
            )

            biv_continua('qtd_filhos', 4, df_)
            biv_continua('idade', 10, df_)
            biv_continua('log_tempo_emprego', 10, df_)
            biv_continua('log_renda', 10, df_)

            st.markdown(
                """
                ### Conclusões Gerais

                - **Weight of Evidence (WoE):** Uma métrica poderosa para entender a relação entre variáveis explicativas e a variável resposta.
                - **Intervalos de Confiança:** Adicionar limites superior e inferior aumenta a confiança nas interpretações.
                - **Análise de Frequência:** Garantir que todas as categorias/faixas tenham um volume significativo é crucial para estabilidade do modelo.

                **Recomendações:**
                - Evitar categorias/faixas com volumes baixos para minimizar vieses.
                - Analisar padrões de WoE para identificar variáveis preditoras mais importantes.
                - Considerar remodelagem de variáveis que apresentem instabilidade nas categorias/faixas.
                """
            )

        def calcular_iv_e_plotar(df, coluna, target='mau', bins=5):
            # Verifica se a coluna é numérica e discretiza caso seja
            if pd.api.types.is_numeric_dtype(df[coluna]):
                df['coluna_binned'] = pd.qcut(df[coluna], q=bins, duplicates='drop')
            else:
                df['coluna_binned'] = df[coluna]

            # Agrupar e calcular a frequência, taxa de evento e outros cálculos para o IV
            group = df.groupby(['data_ref', 'coluna_binned', target]).count()['tipo_residencia'].to_frame().rename(columns={"tipo_residencia": "Freq"})
            tab = group.unstack()['Freq']
            tab['N'] = tab.sum(axis=1)
            tab['tx_evento'] = tab[1] / tab.N
            tab['pct_evento'] = tab[1] / tab[1].groupby(level=0).sum()
            tab['pct_nao_evento'] = tab[0] / tab[0].groupby(level=0).sum()
            tab['WOE'] = np.log(tab['pct_evento'] / tab['pct_nao_evento'])

            # Cálculo do IV
            iv = ((tab['pct_evento'] - tab['pct_nao_evento']) * tab['WOE']).groupby(level=0).sum()

            # Plot do IV ao longo das safras
            fig, ax = plt.subplots(figsize=(10, 6))
            x = iv.index
            a = [.02] * len(iv.index)
            b = [.1] * len(iv.index)
            c = [.3] * len(iv.index)
            d = [.5] * len(iv.index)
            e = [.6] * len(iv.index)

            # Faixas de cores para valores de IV
            ax.fill_between(iv.index, a, color='grey', alpha=.2)
            ax.fill_between(iv.index, a, b, color='orange', alpha=.1)  # Sem o label aqui
            ax.fill_between(iv.index, b, c, color='green', alpha=.1)   # Sem o label aqui
            ax.fill_between(iv.index, c, d, color='blue', alpha=.1)    # Sem o label aqui
            ax.fill_between(iv.index, d, e, color='purple', alpha=.1)  # Sem o label aqui


            ax.plot(iv, marker='o', color='black')  # Linha do IV ao longo do tempo
            ax.set_title(f"Information Value para {coluna.capitalize()} ao longo das safras")
            ax.set_ylabel("Information Value (IV)")
            ax.set_xlabel("Safra")
            ax.legend(loc='upper left')

            st.pyplot(plt)

        with tab5:
            # Introdução ao IV
            st.markdown("## **Análise de Information Value (IV)**")

            st.markdown("""
            ### **O que é Information Value (IV)?**

            O **Information Value (IV)** é uma métrica amplamente utilizada em análises de risco de crédito para medir o poder de discriminação de uma variável explicativa em relação a uma variável resposta binária (por exemplo, 'bom' ou 'mau pagador'). Em outras palavras, ele indica quão bem uma variável consegue separar eventos e não-eventos.
            """)

            # Benefícios do IV
            st.info("""
            ### **Por que usar IV?**

            - Selecionar variáveis relevantes para modelos preditivos.
            - Identificar variáveis com baixo poder preditivo, que podem ser descartadas.
            - Avaliar a qualidade da segmentação dos dados em faixas ou categorias.
            """)

            # Cálculo do IV
            st.markdown("""
            ### **Como o IV é calculado?**

            O IV é baseado no conceito de **Weight of Evidence (WoE)**, que mede a força de discriminação de uma variável em cada faixa ou categoria.
            """)

            # Fórmula do WoE
            st.markdown("A fórmula do WoE é:")
            st.latex(r'''
            WOE = \ln\left(\frac{\%\text{ de eventos na categoria}}{\%\text{ de não-eventos na categoria}}\right)
            ''')

            st.markdown("""
            O IV, por sua vez, é calculado como o somatório ponderado das diferenças entre as proporções de eventos e não-eventos para cada faixa ou categoria:
            """)

            # Fórmula do IV
            st.latex(r'''
            IV = \sum_{k}\left(\%\text{ de eventos}_k - \%\text{ de não-eventos}_k\right) \times WOE_k
            ''')

            st.markdown("""
            Onde:
            - \( k \): Cada faixa ou categoria da variável explicativa.
            - \( \%\text{ de eventos}_k \): Proporção de eventos na categoria \( k \).
            - \( \%\text{ de não-eventos}_k \): Proporção de não-eventos na categoria \( k \).

            Essas métricas ajudam a entender o poder preditivo de uma variável e a identificar quais são mais relevantes para o modelo.
            """)

            # Interpretação do IV
            st.markdown("""
            ### **Como interpretar os valores de IV?**

            Os valores de IV podem ser interpretados como segue:
            """)

            st.markdown("""
            | **Intervalo de IV** | **Interpretação**          |
            |----------------------|----------------------------|
            | IV < 0.02            | Sem poder preditivo       |
            | 0.02 ≤ IV < 0.1      | Poder preditivo fraco     |
            | 0.1 ≤ IV < 0.3       | Poder preditivo médio     |
            | 0.3 ≤ IV < 0.5       | Poder preditivo forte     |
            | IV ≥ 0.5             | Poder preditivo muito forte |
            """)

            # Sobre os gráficos
            st.info("""
            ### **O que os gráficos mostram?**

            - **Relevância da variável:** Se o IV permanece elevado ao longo do tempo, indica consistência no poder preditivo.
            - **Estabilidade temporal:** Oscilações grandes podem indicar que a variável não é estável entre períodos.
            - **Segmentação ideal:** Avaliar se a discretização ou as categorias da variável estão apropriadas.
            """)

            # Objetivo da análise
            st.markdown("""
            ### **Objetivo da análise**

            - Identificar as variáveis mais relevantes para prever o comportamento da variável resposta.
            - Analisar a estabilidade temporal do poder preditivo das variáveis.
            - Descobrir potenciais melhorias no agrupamento ou discretização dos dados.

            Os gráficos abaixo apresentam os resultados do IV para as variáveis selecionadas.
            """)

            colunas = df_.columns
            colunas.drop(['mau','index','bom'])
            for var in colunas:
                calcular_iv_e_plotar(df_,var)
        
        with tab6:
            df__ =  pd.read_feather(trade_history_file)
            df = df__.copy()
            st.markdown("""
                <div style="text-align: center;">
                    <h3>Enquanto o modelo é carregado...</h3>
                    <p>Aguarde, esse processo pode levar alguns segundos!</p>
                    <img src="https://i.giphy.com/pFZTlrO0MV6LoWSDXd.webp" alt="Aguarde..." width="400"/>
                    <p>Por favor, aguarde...</p>
                </div>
            """, unsafe_allow_html=True)
            st.divider()
            st.markdown("""
                <div style="text-align: center;">
                    <h3>Modeling</h3>
                </div>
            """, unsafe_allow_html=True)


            shape, loc, scale = lognorm.fit(df['tempo_emprego'].dropna(), floc=0)
            missing_count = df['tempo_emprego'].isna().sum()
            imputed_values = lognorm.rvs(shape, loc=loc, scale=scale, size=missing_count)
            df.loc[df['tempo_emprego'].isna(), 'tempo_emprego'] = imputed_values
            df['tempo_emprego'] = pd.qcut(df['tempo_emprego'],q=5)
            bins = [-100,2000,4000,8000,10000,15000,20000,25000,30000,1000000000]
            labels = ['até 2000','entre 2k e 4k','entre 4k e 8k','entre 8k e 10k', 'entre 10k e 15k', 'entre 15k e 20k', 'entre 20k e 25k','entre 25k e 30k' ,'acima de 30k']
            df['renda'] = pd.cut(df['renda'],bins=bins,labels=labels)
            df['Mes'] = pd.to_datetime(df['data_ref']).dt.month_name()
            bins = [0,30,40,50,60,200]
            labels = ['Menos de 30','entre 31 e 40','entre 41 e 50', 'entre 51 e 60','acima de 60']
            df['idade'] = pd.cut(df['idade'],bins=bins,labels=labels)

            lb = LabelEncoder()
            variaveis = df.select_dtypes(exclude='number').drop(columns=['data_ref','mau',]).columns
            for var in variaveis:
                df[var] = lb.fit_transform(df[var])


            target,data = df['mau'],df['data_ref']
            df_ = df.copy()
            df_.drop(columns=['mau','data_ref','index'],inplace=True)

            scale = StandardScaler()
            labels = df_.columns.tolist()
            df_pad = pd.DataFrame(scale.fit_transform(df_),columns=labels)

            df_pad['data_ref'] = data
            df_pad['mau'] = target
            #---------------------------


            df_part = df_pad.loc[1:25001,:]
            exp_clf101 = setup(data = df_part, target = 'mau', session_id=123)
            st.write(exp_clf101)
            #tenho que exibir isso
            gbc = create_model('gbc')
            tuned_gbc = tune_model(gbc,fold=5)

            col1,col2 = st.columns(2)
            with col1:
                st.write("### Model Gradient Boosting")
                st.write(gbc)
            
            with col2:
                st.write("### Tuned Gradient Boosting")
                st.write(tuned_gbc)
            st.divider()

            col1,col2 = st.columns(2)
            with col1:
                plot_model(tuned_gbc, plot='auc', save=True)
                st.image("AUC.png", caption="Curva ROC (AUC) - A área sob a curva", use_column_width=True)
                plot_model(tuned_gbc, plot='pr', save=True)
                st.image("Precision Recall.png", caption="Gráfico Precision-Recall", use_column_width=True)
            with col2:
                plot_model(tuned_gbc, plot='feature', save=True)
                st.image("Feature Importance.png", caption="Importância das Características", use_column_width=True)
                plot_model(tuned_gbc, plot='confusion_matrix', save=True)
                st.image("Confusion Matrix.png", caption="Matriz de Confusão", use_column_width=True)
            
            st.info("""
                ### Gráfico AUC (Curva de Característica Operacional do Receptor)
                A AUC (Área sob a Curva) é uma métrica de desempenho amplamente usada em modelos de classificação. Ela representa a capacidade do modelo em distinguir entre classes. 
                Quanto maior a AUC, melhor é o modelo para classificar as observações corretamente. A AUC varia entre 0 e 1:
                - **AUC > 0.9**: Excelente.
                - **0.7 < AUC <= 0.9**: Bom.
                - **0.5 < AUC <= 0.7**: Modesto.
                - **AUC <= 0.5**: O modelo não é melhor do que uma escolha aleatória.
                """)

            st.info("""
                ### Gráfico Precision-Recall (Precisão vs Revocação)
                O gráfico Precision-Recall mostra a relação entre precisão e revocação para diferentes limiares de classificação. Ele é particularmente útil quando se lida com dados desbalanceados. 
                - **Precisão (Precision)**: A proporção de predições positivas corretas em relação ao total de predições positivas feitas.
                - **Revocação (Recall)**: A proporção de verdadeiros positivos em relação ao total de positivos reais.
                Idealmente, você quer maximizar tanto a precisão quanto a revocação, o que resulta em uma curva PR mais alta.
                """)

            st.info("""
                ### Gráfico de Importância das Características
                Este gráfico mostra a importância relativa de cada característica (variável) no modelo de classificação. Características mais importantes são aquelas que contribuem mais para a predição do modelo. 
                - As variáveis com alta importância devem ser consideradas para tomadas de decisão.
                - Variáveis com baixa importância podem ser descartadas ou requerem mais análise.
                """)


            st.info("""
                ### Matriz de Confusão
                A matriz de confusão fornece uma visão detalhada do desempenho do modelo de classificação, mostrando quantas predições foram corretas e incorretas para cada classe.
                - **Verdadeiros Positivos (TP)**: O modelo previu corretamente a classe positiva.
                - **Falsos Positivos (FP)**: O modelo previu a classe positiva quando deveria ter sido negativa.
                - **Verdadeiros Negativos (TN)**: O modelo previu corretamente a classe negativa.
                - **Falsos Negativos (FN)**: O modelo previu a classe negativa quando deveria ter sido positiva.
                Ela é útil para calcular métricas como acurácia, precisão, revocação e F1-score.
                """)
            st.divider()
            final_gbc = finalize_model(tuned_gbc)
            st.write('## PIPELINE DO MODELO: ')
            save_model(final_gbc,'Final_GB_Model')
            st.write(save_model(final_gbc,'Final_GB_Model'))

if __name__ == '__main__':
    main()