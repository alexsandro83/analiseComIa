import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re

# Carregar os dados
df = pd.read_csv('todos_ate_05-10-25.csv', sep=';')

print(f"Dataset shape: {df.shape}")

# Função para extrair time principal do Stat
def extract_main_team(stat):
    if pd.isna(stat):
        return "Unknown"
    
    stat_str = str(stat)
    
    # Padrões para encontrar o time principal
    patterns = [
        r'^([A-Za-z0-9\s\-\']+) have Won their last',
        r'^([A-Za-z0-9\s\-\']+) have had BTTS in their last',
        r'^([A-Za-z0-9\s\-\']+) have had Over',
        r'^([A-Za-z0-9\s\-\']+) have Lost their last',
        r'^([A-Za-z0-9\s\-\']+) have Drew their last',
        r'^([A-Za-z0-9\s\-\']+) have drawn their last'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, stat_str)
        if match:
            return match.group(1).strip()
    
    return "Unknown"

# Função para extrair time adversário do Next Match
def extract_opponent_team(next_match):
    if pd.isna(next_match):
        return "Unknown"
    
    next_match_str = str(next_match)
    
    # Padrões para encontrar o adversário
    patterns = [
        r'vs\s+([A-Za-z0-9\s\-\']+)$',
        r'@\s+([A-Za-z0-9\s\-\']+)$',
        r'Away vs\s+([A-Za-z0-9\s\-\']+)$',
        r'Home vs\s+([A-Za-z0-9\s\-\']+)$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, next_match_str)
        if match:
            return match.group(1).strip()
    
    return "Unknown"

# Pré-processamento dos dados
def preprocess_data(df):
    df_clean = df.copy()
    
    # 1. Extrair times
    print("Extraindo times principais e adversários...")
    df_clean['Main_Team'] = df_clean['Stat'].apply(extract_main_team)
    df_clean['Opponent_Team'] = df_clean['Next Match'].apply(extract_opponent_team)
    
    print(f"Times principais únicos: {df_clean['Main_Team'].nunique()}")
    print(f"Times adversários únicos: {df_clean['Opponent_Team'].nunique()}")
    
    # 2. Criar ID único do jogo
    df_clean['Game_ID'] = (
        df_clean['Main_Team'] + ' vs ' + df_clean['Opponent_Team'] + ' - ' + 
        df_clean['Date'].astype(str)
    )
    
    print(f"Jogos únicos: {df_clean['Game_ID'].nunique()}")
    
    # 3. Codificar variáveis categóricas
    label_encoders = {}
    categorical_columns = ['League', 'Stat', 'Next Match', 'Main_Team', 'Opponent_Team']
    
    for col in categorical_columns:
        if col in df_clean.columns:
            le = LabelEncoder()
            df_clean[col + '_encoded'] = le.fit_transform(df_clean[col].astype(str))
            label_encoders[col] = le
    
    # 4. Processar Odds
    df_clean['Odds'] = pd.to_numeric(df_clean['Odds'], errors='coerce')
    df_clean['Odds'].fillna(df_clean['Odds'].mean(), inplace=True)
    
    # 5. Extrair features do Stat
    df_clean['Stat_Type'] = df_clean['Stat'].apply(lambda x: 
        'Win' if 'Won' in str(x) else
        'BTTS' if 'BTTS' in str(x) else
        'Over' if 'Over' in str(x) else
        'Lost' if 'Lost' in str(x) else
        'Draw' if 'Drew' in str(x) else 'Other'
    )
    
    # 6. Extrair número de jogos da sequência
    def extract_streak_length(stat):
        if pd.isna(stat):
            return 6
        stat_str = str(stat)
        numbers = [int(s) for s in stat_str.split() if s.isdigit()]
        return numbers[0] if numbers else 6
    
    df_clean['Streak_Length'] = df_clean['Stat'].apply(extract_streak_length)
    
    # 7. Identificar se é sequência em Casa ou Fora
    df_clean['Is_Home_Streak'] = df_clean['Stat'].apply(
        lambda x: 1 if 'home' in str(x).lower() else 0
    )
    
    df_clean['Is_Away_Streak'] = df_clean['Stat'].apply(
        lambda x: 1 if 'away' in str(x).lower() else 0
    )
    
    # 8. Identificar local do próximo jogo
    df_clean['Next_Match_Location'] = df_clean['Next Match'].apply(
        lambda x: 'Home' if 'Home' in str(x) else 'Away' if 'Away' in str(x) else 'Neutral'
    )
    
    # 9. Extrair informações da DATA
    df_clean['Date_dt'] = pd.to_datetime(df_clean['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
    df_clean['Season_Month'] = df_clean['Date_dt'].dt.month
    df_clean['Is_Weekend'] = df_clean['Date_dt'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # 10. Codificar variáveis adicionais
    le_stat = LabelEncoder()
    df_clean['Stat_Type_encoded'] = le_stat.fit_transform(df_clean['Stat_Type'])
    label_encoders['Stat_Type'] = le_stat
    
    le_location = LabelEncoder()
    df_clean['Next_Match_Location_encoded'] = le_location.fit_transform(df_clean['Next_Match_Location'])
    label_encoders['Next_Match_Location'] = le_location
    
    # 11. Codificar variável target
    le_target = LabelEncoder()
    df_clean['Situacao_encoded'] = le_target.fit_transform(df_clean['Situacao'])
    label_encoders['Situacao'] = le_target
    
    return df_clean, label_encoders

# Aplicar pré-processamento
df_processed, label_encoders = preprocess_data(df)

print(f"\nDataset após pré-processamento: {df_processed.shape}")

# Mostrar alguns exemplos de times extraídos
print("\n🔍 EXEMPLOS DE TIMES EXTRAÍDOS:")
sample_data = df_processed[['Stat', 'Main_Team', 'Next Match', 'Opponent_Team', 'Game_ID']].head(10)
for _, row in sample_data.iterrows():
    print(f"Stat: {row['Stat'][:50]}...")
    print(f"Time Principal: {row['Main_Team']}")
    print(f"Adversário: {row['Opponent_Team']}")
    print(f"Game ID: {row['Game_ID'][:60]}...")
    print("-" * 80)

# Selecionar features RELEVANTES
feature_columns = [
    'Odds', 
    'Streak_Length',                    # Tamanho da sequência
    'Is_Home_Streak',                   # Sequência em casa
    'Is_Away_Streak',                   # Sequência fora
    'League_encoded',                   # Liga
    'Stat_Type_encoded',                # Tipo de estatística
    'Main_Team_encoded',                # Time principal
    'Opponent_Team_encoded',            # Time adversário
    'Next_Match_Location_encoded',      # Local do próximo jogo
    'Season_Month',                     # Mês da temporada
    'Is_Weekend'                        # Final de semana
]

# Verificar se todas as colunas existem
available_features = [col for col in feature_columns if col in df_processed.columns]
print(f"\n🎯 FEATURES SELECIONADAS ({len(available_features)}):")
for i, feature in enumerate(available_features, 1):
    print(f"{i:2d}. {feature}")

X = df_processed[available_features]
y = df_processed['Situacao_encoded']

print(f"\n📊 Shape X: {X.shape}, Shape y: {y.shape}")
print(f"🎲 Distribuição das classes: {np.bincount(y)}")
print(f"   VERDADEIRO: {np.bincount(y)[1]}, FALSO: {np.bincount(y)[0]}")

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📚 Treino: {X_train.shape}, Teste: {X_test.shape}")

# Treinar o modelo
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

print("\n🤖 Treinando o modelo...")
model.fit(X_train, y_train)

# Avaliar o modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n=== 📈 AVALIAÇÃO DO MODELO ===")
print(f"🎯 Acurácia: {accuracy:.4f}")
print(f"\n📋 Relatório de Classificação:")
print(classification_report(y_test, y_pred, 
                          target_names=label_encoders['Situacao'].classes_))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🔥 IMPORTÂNCIA DAS FEATURES:")
for _, row in feature_importance.iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# Salvar o modelo
model_data = {
    'model': model,
    'label_encoders': label_encoders,
    'feature_columns': available_features,
    'metadata': {
        'accuracy': accuracy,
        'n_features': len(available_features),
        'n_samples': len(df_processed),
        'class_distribution': np.bincount(y),
        'features_importance': feature_importance.to_dict()
    }
}

joblib.dump(model_data, 'modelo_sequencias_com_times.joblib')

print(f"\n✅ Modelo salvo como 'modelo_sequencias_com_times.joblib'")

# Função para fazer previsões
def predict_sequence(league, stat, next_match, odds, date):
    """
    Função para fazer previsões com o modelo treinado
    """
    # Carregar modelo
    model_data = joblib.load('modelo_sequencias_com_times.joblib')
    model = model_data['model']
    label_encoders = model_data['label_encoders']
    feature_columns = model_data['feature_columns']
    
    # Preparar dados de entrada
    input_data = {}
    
    # 1. Extrair times
    main_team = extract_main_team(stat)
    opponent_team = extract_opponent_team(next_match)
    
    # 2. Odds
    input_data['Odds'] = float(odds) if odds else 1.5
    
    # 3. Extrair informações do Stat
    numbers = [int(s) for s in stat.split() if s.isdigit()]
    input_data['Streak_Length'] = numbers[0] if numbers else 6
    
    input_data['Is_Home_Streak'] = 1 if 'home' in stat.lower() else 0
    input_data['Is_Away_Streak'] = 1 if 'away' in stat.lower() else 0
    
    # Stat Type
    stat_type = 'Win' if 'Won' in stat else 'BTTS' if 'BTTS' in stat else 'Over' if 'Over' in stat else 'Lost' if 'Lost' in stat else 'Draw' if 'Drew' in stat else 'Other'
    input_data['Stat_Type_encoded'] = label_encoders['Stat_Type'].transform([stat_type])[0]
    
    # 4. Codificar variáveis categóricas
    try:
        input_data['League_encoded'] = label_encoders['League'].transform([league])[0]
    except:
        input_data['League_encoded'] = 0
    
    try:
        input_data['Main_Team_encoded'] = label_encoders['Main_Team'].transform([main_team])[0]
    except:
        input_data['Main_Team_encoded'] = 0
    
    try:
        input_data['Opponent_Team_encoded'] = label_encoders['Opponent_Team'].transform([opponent_team])[0]
    except:
        input_data['Opponent_Team_encoded'] = 0
    
    # 5. Local do próximo jogo
    next_location = 'Home' if 'Home' in str(next_match) else 'Away' if 'Away' in str(next_match) else 'Neutral'
    input_data['Next_Match_Location_encoded'] = label_encoders['Next_Match_Location'].transform([next_location])[0]
    
    # 6. Informações da data
    date_obj = pd.to_datetime(date, format='%d/%m/%Y %H:%M')
    input_data['Season_Month'] = date_obj.month
    input_data['Season_Month'] = input_data['Season_Month'] - 8 if input_data['Season_Month'] >= 9 else input_data['Season_Month'] + 4
    input_data['Is_Weekend'] = 1 if date_obj.dayofweek in [5, 6] else 0
    
    # Criar array de features
    features = np.array([[input_data[col] for col in feature_columns]])
    
    # Fazer previsão
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    # Decodificar resultado
    result = label_encoders['Situacao'].inverse_transform([prediction])[0]
    prob_verdadeiro = probability[1] if result == 'VERDADEIRO' else probability[0]
    
    return {
        'prediction': result,
        'probability': max(probability),
        'confidence': prob_verdadeiro,
        'main_team': main_team,
        'opponent_team': opponent_team,
        'game_id': f"{main_team} vs {opponent_team} - {date}",
        'features_used': feature_columns
    }

# Testar a função de previsão
print(f"\n🧪 TESTE DE PREVISÃO:")
test_prediction = predict_sequence(
    league='La Liga',
    stat='Real Madrid have Won their last 6 league matches',
    next_match='Away vs Ath Madrid',
    odds=2.25,
    date='27/09/2025 11:15'
)

print(f"🏆 Jogo: {test_prediction['main_team']} vs {test_prediction['opponent_team']}")
print(f"🎯 Previsão: {test_prediction['prediction']}")
print(f"📊 Confiança: {test_prediction['confidence']:.2%}")
print(f"🔍 Probabilidade: {test_prediction['probability']:.2%}")

print(f"\n🎉 MODELO CRIADO COM SUCESSO!")
print(f"📈 Acurácia: {accuracy:.2%}")
print(f"🔧 Features utilizadas: {len(available_features)}")
print(f"📚 Amostras de treino: {X_train.shape[0]}")
print(f"💾 Modelo salvo: 'modelo_sequencias_com_times.joblib'")