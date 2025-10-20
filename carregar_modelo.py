import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Carregar o modelo salvo
modelo_salvo = joblib.load('modelo_apostas_evolutivo.joblib')
modelo = modelo_salvo['modelo']
encoder = modelo_salvo['encoder']

# Carregar sua base atual
df = pd.read_csv('todos_ate_05-10-25.csv', delimiter=';')

# Função para preparar os dados NOVOS
def preparar_features(df):
    # 1. Odds (já está na base)
    df['Odds'] = pd.to_numeric(df['Odds'], errors='coerce')
    
    # 2. Tamanho_Streak - extrair do campo 'Stat'
    def extrair_tamanho_streak(stat):
        import re
        match = re.search(r'last\s+(\d+)', stat)
        return int(match.group(1)) if match else 6  # default 6 se não encontrar
    
    df['Tamanho_Streak'] = df['Stat'].apply(extrair_tamanho_streak)
    
    # 3. Tipo_Estatistica - categorizar o tipo de estatística
    def categorizar_estatistica(stat):
        if 'BTTS' in stat:
            return 'BTTS'
        elif 'Won' in stat:
            return 'Vitoria'
        elif 'Over 2.5' in stat:
            return 'Over_2_5'
        elif 'Lost' in stat:
            return 'Derrota'
        elif 'Drew' in stat:
            return 'Empate'
        else:
            return 'Outro'
    
    df['Tipo_Estatistica'] = df['Stat'].apply(categorizar_estatistica)
    
    # 4. Local_Jogo - extrair de 'Next Match'
    def extrair_local_jogo(next_match):
        if 'Home' in next_match:
            return 'Home'
        elif 'Away' in next_match:
            return 'Away'
        else:
            return 'Neutro'
    
    df['Local_Jogo'] = df['Next Match'].apply(extrair_local_jogo)
    
    # 5. Liga_Categoria - categorizar a liga
    # (usando a mesma lógica do treinamento original)
    ligas_principais = ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Liga MX', 'Liga Pro']
    
    def categorizar_liga(league):
        if any(liga in league for liga in ligas_principais):
            return 'Principal'
        else:
            return 'Outras'
    
    df['Liga_Categoria'] = df['League'].apply(categorizar_liga)
    
    return df

# Preparar os dados
df_preparado = preparar_features(df)

# Selecionar apenas as features que o modelo espera
features_finais = df_preparado[['Odds', 'Tamanho_Streak', 'Tipo_Estatistica', 'Local_Jogo', 'Liga_Categoria']]

# Codificar variáveis categóricas (usando o mesmo encoder do treinamento)
try:
    features_encoded = encoder.transform(features_finais)
except:
    # Se o encoder não funcionar, fazer one-hot manual
    features_encoded = pd.get_dummies(features_finais, 
                                    columns=['Tipo_Estatistica', 'Local_Jogo', 'Liga_Categoria'])
    
    # Garantir que temos as mesmas colunas que o modelo espera
    expected_features = modelo.feature_names_in_
    for col in expected_features:
        if col not in features_encoded.columns:
            features_encoded[col] = 0
    
    features_encoded = features_encoded[expected_features]

# Fazer previsões
previsoes = modelo.predict(features_encoded)
probabilidades = modelo.predict_proba(features_encoded)

# Adicionar resultados ao DataFrame
df_preparado['Previsao'] = previsoes
df_preparado['Probabilidade_VERDADEIRO'] = probabilidades[:, 1]  # Probabilidade da classe positiva
df_preparado['Probabilidade_FALSO'] = probabilidades[:, 0]       # Probabilidade da classe negativa

# Mostrar resultados
print("🔮 PREVISÕES PARA AS PRÓXIMAS PARTIDAS:")
print("=" * 80)

resultados_finais = df_preparado[['League', 'Stat', 'Next Match', 'Odds', 'Date', 
                                 'Previsao', 'Probabilidade_VERDADEIRO']].copy()

resultados_finais['Previsao_Texto'] = resultados_finais['Previsao'].map({1: 'VERDADEIRO', 0: 'FALSO'})

# Ordenar por probabilidade de acerto
resultados_finais = resultados_finais.sort_values('Probabilidade_VERDADEIRO', ascending=False)

print(resultados_finais.head(20).to_string(index=False))

# Estatísticas gerais
print(f"\n📈 ESTATÍSTICAS DAS PREVISÕES:")
print(f"Total de jogos analisados: {len(resultados_finais)}")
print(f"Previsões VERDADEIRO: {(resultados_finais['Previsao'] == 1).sum()}")
print(f"Previsões FALSO: {(resultados_finais['Previsao'] == 0).sum()}")
print(f"Confiança média: {resultados_finais['Probabilidade_VERDADEIRO'].mean():.2%}")

# Salvar resultados
resultados_finais.to_csv('previsoes_futuras.csv', index=False)
print(f"\n💾 Resultados salvos em 'previsoes_futuras.csv'")