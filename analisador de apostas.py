import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
from datetime import datetime
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class AnalisadorApostasEvolutivo:
    def __init__(self, base_treino_path=None, base_futuros_path=None, modelo_path='modelo_apostas.joblib'):
        self.base_treino_path = base_treino_path
        self.base_futuros_path = base_futuros_path
        self.modelo_path = modelo_path
        self.model = None
        self.encoder = LabelEncoder()
        self.jogos_complementares = {}
        self.features_para_treino = ['Odds', 'Tamanho_Streak', 'Tipo_Estatistica', 'Local_Jogo', 'Liga_Categoria']
        
    def carregar_dados(self, csv_path):
    
        try:
            # Primeira tentativa: encoding utf-8
            return pd.read_csv(csv_path, delimiter=';', encoding='utf-8-sig')
        except pd.errors.ParserError as e:
            print(f"❌ Erro de parsing detectado: {e}")
            print("🔧 Tentando corrigir automaticamente...")
            
            # Lê o arquivo linha por linha para identificar o problema
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                linhas = f.readlines()
            
            # Identifica a linha problemática
            linha_problema = 135  # baseado no erro
            if len(linhas) >= linha_problema:
                print(f"📝 Linha {linha_problema} problemática: {linhas[linha_problema-1]}")
            
            # Tenta carregar com tratamento de erro mais flexível
            try:
                df = pd.read_csv(csv_path, delimiter=';', encoding='utf-8-sig', 
                            on_bad_lines='skip',  # Pula linhas problemáticas
                            engine='python')
                print(f"✅ Carregado com {len(df)} linhas após correção automática")
                return df
            except:
                # Última tentativa: carrega manualmente
                print("🔄 Carregando manualmente...")
                dados_corrigidos = []
                cabecalho = linhas[0].strip().split(';')
                
                for i, linha in enumerate(linhas[1:], 2):  # Começa da linha 2 (pula cabeçalho)
                    campos = linha.strip().split(';')
                    if len(campos) == len(cabecalho):
                        dados_corrigidos.append(campos)
                    else:
                        print(f"⚠️  Linha {i} ignorada: número de campos incorreto")
                
                df = pd.DataFrame(dados_corrigidos, columns=cabecalho)
                print(f"✅ Carregamento manual: {len(df)} linhas processadas")
                return df
                
        except UnicodeDecodeError:
            try:
                return pd.read_csv(csv_path, delimiter=';', encoding='latin-1')
            except:
                return pd.read_csv(csv_path, delimiter=';', encoding='iso-8859-1')
    def treinar_modelo_evolutivo(self, forcar_retreino=False):
        """Treinar ou atualizar modelo com dados mais recentes"""
        print("🔄 INICIANDO TREINAMENTO EVOLUTIVO...")
        
        # Verificar se modelo existe e se deve atualizar
        if os.path.exists(self.modelo_path) and not forcar_retreino:
            print("📁 Modelo existente encontrado. Verificando necessidade de atualização...")
            if not self._verificar_necessidade_atualizacao():
                print("✅ Modelo atualizado. Carregando modelo existente...")
                self.carregar_modelo()
                return True
        
        if self.base_treino_path is None:
            print("❌ Caminho da base de treino não especificado")
            return False
            
        # Carregar e preparar dados
        self.df_treino = self.carregar_dados(self.base_treino_path)
        self._preparar_dados_treino()
        
        if len(self.df_treino_limpo) < 10:
            print("❌ Dados insuficientes para treinamento")
            return False
        
        # Treinar novo modelo
        print("🎯 Treinando novo modelo com dados atualizados...")
        X = self.df_treino_limpo[self.features_para_treino].copy()
        X['Tipo_Estatistica'] = self.encoder.fit_transform(X['Tipo_Estatistica'])
        X['Local_Jogo'] = LabelEncoder().fit_transform(X['Local_Jogo'])
        X['Liga_Categoria'] = LabelEncoder().fit_transform(X['Liga_Categoria'])
        
        y = self.df_treino_limpo['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=12, min_samples_split=5)
        self.model.fit(X_train, y_train)
        
        accuracy = self.model.score(X_test, y_test)
        self.acuracia_modelo = accuracy
        
        # Salvar modelo
        self._salvar_modelo()
        
        print(f"✅ Modelo treinado com sucesso! Acurácia: {accuracy:.2%}")
        print(f"📊 Total de amostras de treino: {len(self.df_treino_limpo)}")
        
        return True
    
    def _verificar_necessidade_atualizacao(self):
        """Verificar se base de treino tem novos dados"""
        if not os.path.exists(self.modelo_path) or self.base_treino_path is None:
            return True
            
        # Carregar dados atuais
        df_atual = self.carregar_dados(self.base_treino_path)
        df_atual_limpo = df_atual[df_atual['Situacao'].notna() & (df_atual['Situacao'] != '')]
        
        # Carregar info do modelo salvo
        try:
            modelo_data = joblib.load(self.modelo_path)
            amostras_anteriores = modelo_data.get('amostras_treino', 0)
            
            # Se tem pelo menos 10% mais dados, retreinar
            if len(df_atual_limpo) > amostras_anteriores * 1.1:
                print(f"📈 Novos dados detectados: {len(df_atual_limpo)} vs {amostras_anteriores}")
                return True
        except:
            return True
            
        return False
    
    def _preparar_dados_treino(self):
        """Preparar dados para treino"""
        # Converter datas
        def corrigir_data(date_str):
            try:
                date_with_year = f"{date_str} 2025"
                return pd.to_datetime(date_with_year, format='%A, %d %B %H:%M %Y', errors='coerce')
            except:
                return pd.NaT
        
        self.df_treino['Date'] = self.df_treino['Date'].apply(corrigir_data)
        self.df_treino['Odds'] = pd.to_numeric(self.df_treino['Odds'], errors='coerce')
        
        # Target
        situacao_map = {'VERDADEIRO': 1, 'Verdadeiro': 1, 'FALSO': 0, 'Falso': 0}
        self.df_treino['Target'] = self.df_treino['Situacao'].map(situacao_map).fillna(-1)
        
        # Features
        self.df_treino['Tipo_Estatistica'] = self.df_treino['Stat'].apply(self._classificar_estatistica)
        self.df_treino['Tamanho_Streak'] = self.df_treino['Stat'].apply(self._extrair_streak)
        self.df_treino['Local_Jogo'] = self.df_treino['Next Match'].apply(self._extrair_local)
        self.df_treino['Liga_Categoria'] = self.df_treino['League'].apply(self._classificar_liga)
        
        self.df_treino_limpo = self.df_treino[self.df_treino['Target'] != -1].copy()
    
    def _classificar_estatistica(self, stat):
        """Classificar tipo de estatística"""
        stat_str = str(stat).lower()
        if 'won' in stat_str:
            return 'VITORIA'
        elif 'lost' in stat_str:
            return 'DERROTA'
        elif 'drew' in stat_str:
            return 'EMPATE'
        elif 'btts' in stat_str:
            return 'BTTS'
        elif 'over 2.5' in stat_str:
            return 'OVER_2.5'
        else:
            return 'OUTRO'
    
    def _extrair_streak(self, stat):
        """Extrair tamanho do streak"""
        matches = re.findall(r'last (\d+)', str(stat))
        return int(matches[0]) if matches else 1
    
    def _extrair_local(self, next_match):
        """Extrair local do jogo"""
        match_str = str(next_match).lower()
        if 'home' in match_str:
            return 'CASA'
        elif 'away' in match_str:
            return 'FORA'
        else:
            return 'NEUTRO'
    
    def _classificar_liga(self, league):
        """Classificar liga por confiabilidade"""
        league_str = str(league).lower()
        if any(x in league_str for x in ['nwsl', 'women', 'norway', 'denmark']):
            return 'ALTA_CONFIABILIDADE'
        elif any(x in league_str for x in ['brasil', 'sweden', 'iceland']):
            return 'MEDIA_CONFIABILIDADE'
        else:
            return 'BAIXA_CONFIABILIDADE'
    
    def _calcular_bonus_complementar(self, league, next_match, date):
        """Calcular bônus para jogos complementares"""
        # Implementação básica - retorna 0 por enquanto
        return 0.0
    
    def _calcular_bonus_confiabilidade_liga(self, league):
        """Calcular bônus baseado na confiabilidade da liga"""
        liga_cat = self._classificar_liga(league)
        if liga_cat == 'ALTA_CONFIABILIDADE':
            return 0.08
        elif liga_cat == 'MEDIA_CONFIABILIDADE':
            return 0.05
        else:
            return 0.0
    
    def _classificar_padrao(self, probabilidade):
        """Classificar padrão baseado na probabilidade"""
        if probabilidade > 0.78:
            return 'PADRAO_FORTISSIMO'
        elif probabilidade > 0.68:
            return 'PADRAO_FORTE'
        elif probabilidade > 0.58:
            return 'PADRAO_SOLIDO'
        else:
            return 'PADRAO_REGULAR'
    
    def _classificar_recomendacao(self, probabilidade):
        """Classificar recomendação"""
        if probabilidade > 0.75:
            return 'EXCELENTE'
        elif probabilidade > 0.62:
            return 'BOA'
        else:
            return 'REGULAR'
    
    def _gerar_analise_detalhada(self, prob, previsao, tipo_estatistica, streak, odds, padrao, bonus, liga):
        """Gerar análise detalhada"""
        base = f"{padrao}: {tipo_estatistica} em {streak} jogos (Odds: {odds})"
        if bonus > 0:
            base += f" [BONUS +{bonus:.0%}]"
        base += f" Previsao: {'MANTERA STREAK' if previsao == 1 else 'INTERROMPERA STREAK'}"
        return base
    
    def _preparar_dados_futuros(self, df_futuros):
        """Preparar dados futuros"""
        def corrigir_data(date_str):
            try:
                date_with_year = f"{date_str} 2025"
                return pd.to_datetime(date_with_year, format='%A, %d %B %H:%M %Y', errors='coerce')
            except:
                return pd.NaT
        
        df_futuros['Date'] = df_futuros['Date'].apply(corrigir_data)
        df_futuros['Odds'] = pd.to_numeric(df_futuros['Odds'], errors='coerce')
        df_futuros['Tipo_Estatistica'] = df_futuros['Stat'].apply(self._classificar_estatistica)
        df_futuros['Tamanho_Streak'] = df_futuros['Stat'].apply(self._extrair_streak)
        df_futuros['Local_Jogo'] = df_futuros['Next Match'].apply(self._extrair_local)
        df_futuros['Liga_Categoria'] = df_futuros['League'].apply(self._classificar_liga)
        df_futuros['Time'] = df_futuros['Stat'].apply(lambda x: x.split(' have ')[0] if ' have ' in str(x) else 'TIME_DESCONHECIDO')
    
    def _analise_basica_futuro(self, row):
        """Análise básica para fallback"""
        return {
            'Probabilidade_Sucesso': 0.6,
            'Previsao': 'VERDADEIRO',
            'Padrao': 'PADRAO_REGULAR',
            'Recomendacao': 'REGULAR',
            'Analise_Detalhada': 'Análise básica - dados insuficientes',
            'Bonus_Total': 0.0
        }
    
    def carregar_modelo(self):
        """Carregar modelo treinado"""
        try:
            dados_modelo = joblib.load(self.modelo_path)
            self.model = dados_modelo['modelo']
            self.encoder = dados_modelo['encoder']
            self.acuracia_modelo = dados_modelo['acuracia']
            print(f"✅ Modelo carregado - Acurácia: {self.acuracia_modelo:.2%}")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            return False
    
    def _salvar_modelo(self):
        """Salvar modelo treinado"""
        dados_modelo = {
            'modelo': self.model,
            'encoder': self.encoder,
            'features': self.features_para_treino,
            'acuracia': self.acuracia_modelo,
            'amostras_treino': len(self.df_treino_limpo),
            'data_treinamento': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        joblib.dump(dados_modelo, self.modelo_path)
    
    def gerar_previsoes_futuras(self, output_path='previsoes_inteligentes.csv'):
        """Gerar previsões para jogos futuros com recomendações"""
        if self.model is None:
            print("❌ Modelo não carregado. Execute treinamento primeiro.")
            return
        
        if self.base_futuros_path is None:
            print("❌ Caminho da base de futuros não especificado")
            return
            
        print("🎯 GERANDO PREVISÕES INTELIGENTES...")
        
        # Carregar dados futuros
        df_futuros = self.carregar_dados(self.base_futuros_path)
        self._preparar_dados_futuros(df_futuros)
        
        # Gerar previsões
        previsoes_detalhadas = []
        
        for idx, row in df_futuros.iterrows():
            try:
                previsao = self._analisar_jogo_avancado(row)
                previsoes_detalhadas.append(previsao)
            except Exception as e:
                print(f"⚠️ Erro ao analisar jogo {idx}: {e}")
                previsoes_detalhadas.append(self._analise_basica_futuro(row))
        
        # Adicionar previsões ao DataFrame
        for col in ['Probabilidade_Sucesso', 'Previsao', 'Padrao', 'Recomendacao', 'Analise_Detalhada', 'Bonus_Total']:
            df_futuros[col] = [p[col] for p in previsoes_detalhadas]
        
        # Ordenar por probabilidade e recomendação
        df_futuros['Score_Prioridade'] = df_futuros['Probabilidade_Sucesso'] * df_futuros['Odds']
        df_futuros = df_futuros.sort_values(['Recomendacao', 'Score_Prioridade'], ascending=[False, False])
        
        # Gerar múltiplas recomendadas
        self._gerar_multiplas_recomendadas(df_futuros)
        
        # Salvar resultados
        df_futuros.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')
        print(f"✅ Previsões salvas em: {output_path}")
        print(f"📊 Total de jogos analisados: {len(df_futuros)}")
        print(f"🎯 Jogos EXCELENTES: {len(df_futuros[df_futuros['Recomendacao'] == 'EXCELENTE'])}")
        print(f"👍 Jogos BONS: {len(df_futuros[df_futuros['Recomendacao'] == 'BOA'])}")
        
        return df_futuros
    
    def _analisar_jogo_avancado(self, row):
        """Análise avançada com modelo ML"""
        tipo_estatistica = self._classificar_estatistica(row['Stat'])
        tamanho_streak = self._extrair_streak(row['Stat'])
        local_jogo = self._extrair_local(row['Next Match'])
        liga_categoria = self._classificar_liga(row['League'])
        
        # Bônus
        bonus_complementar = self._calcular_bonus_complementar(row['League'], row['Next Match'], row['Date'])
        bonus_confiabilidade = self._calcular_bonus_confiabilidade_liga(row['League'])
        bonus_streak = min(tamanho_streak * 0.02, 0.10)  # Bônus por streak longo
        bonus_total = bonus_complementar + bonus_confiabilidade + bonus_streak
        
        # Previsão do modelo
        features = pd.DataFrame([{
            'Odds': row['Odds'],
            'Tamanho_Streak': tamanho_streak,
            'Tipo_Estatistica': tipo_estatistica,
            'Local_Jogo': local_jogo,
            'Liga_Categoria': liga_categoria
        }])
        
        features['Tipo_Estatistica'] = self.encoder.transform(features['Tipo_Estatistica'])
        features['Local_Jogo'] = LabelEncoder().fit_transform(features['Local_Jogo'])
        features['Liga_Categoria'] = LabelEncoder().fit_transform(features['Liga_Categoria'])
        
        probabilidade = self.model.predict_proba(features)[0][1]
        previsao = self.model.predict(features)[0]
        
        # Ajustar com bônus
        probabilidade_ajustada = min(probabilidade + bonus_total, 0.95)
        
        # Classificação
        padrao = self._classificar_padrao(probabilidade_ajustada)
        recomendacao = self._classificar_recomendacao(probabilidade_ajustada)
        
        # Análise detalhada
        analise = self._gerar_analise_detalhada(probabilidade_ajustada, previsao, tipo_estatistica, 
                                              tamanho_streak, row['Odds'], padrao, bonus_total, row['League'])
        
        return {
            'Probabilidade_Sucesso': probabilidade_ajustada,
            'Previsao': 'VERDADEIRO' if previsao == 1 else 'FALSO',
            'Padrao': padrao,
            'Recomendacao': recomendacao,
            'Analise_Detalhada': analise,
            'Bonus_Total': bonus_total
        }
    def _gerar_multiplas_recomendadas(self, df_futuros, num_multiplas=20):
        """Gerar múltiplas recomendadas priorizando confiança alta e proximidade temporal"""
        print("\n🎲 GERANDO MÚLTIPLAS RECOMENDADAS:")
        print("="*50)
        
        jogos_excelentes = df_futuros[df_futuros['Recomendacao'] == 'EXCELENTE']
        jogos_bons = df_futuros[df_futuros['Recomendacao'] == 'BOA']
        
        # Combinar todos os jogos recomendados e ordenar por confiança
        todos_jogos = pd.concat([jogos_excelentes, jogos_bons])
        todos_jogos = todos_jogos.sort_values('Probabilidade_Sucesso', ascending=False)
        
        if len(todos_jogos) < 2:
            print("❌ Número insuficiente de jogos para gerar múltiplas")
            return
        
        # Criar lista de jogos com identificadores únicos
        jogos_lista = []
        for idx, row in todos_jogos.iterrows():
            time_casa = row.get('Time_Casa', row.get('Time', ''))
            mercado = row.get('Mercado', row.get('Tipo_Estatistica', ''))
            
            # ADICIONAR INFORMAÇÃO DE DATA/HORA SE DISPONÍVEL
            data_hora = row.get('Data', row.get('Horario', ''))  # Ajuste para suas colunas
            
            jogos_lista.append({
                'id': f"{time_casa}_{mercado}",
                'time': time_casa,
                'mercado': mercado,
                'odds': row.get('Odds', 1.0),
                'confianca': row.get('Probabilidade_Sucesso', 0),
                'data_hora': data_hora,  # ← ADICIONAR PARA PROXIMIDADE TEMPORAL
                'analise': row.get('Analise_Detalhada', '')
            })
        
        # Remover duplicatas exatas
        jogos_unicos = []
        ids_vistos = set()
        for jogo in jogos_lista:
            if jogo['id'] not in ids_vistos:
                ids_vistos.add(jogo['id'])
                jogos_unicos.append(jogo)
        
        if len(jogos_unicos) < 2:
            print("❌ Número insuficiente de jogos únicos para múltiplas")
            return
        
        # Gerar combinações possíveis
        from itertools import combinations
        
        todas_combinacoes = list(combinations(jogos_unicos, 2))
        
        print(f"📊 Gerando {len(todas_combinacoes)} combinações possíveis...")
        
        # Calcular métricas para cada combinação - AQUI ESTÁ O CORAÇÃO DA ALTERAÇÃO!
        multiplas_com_metricas = []
        
        for combo in todas_combinacoes:
            jogo1, jogo2 = combo
            
            # Ordenar para evitar permutações
            jogos_ordenados = sorted([jogo1, jogo2], key=lambda x: x['id'])
            
            # Calcular odd total
            odd1 = jogo1['odds'] if pd.notna(jogo1['odds']) and jogo1['odds'] > 0 else 1.0
            odd2 = jogo2['odds'] if pd.notna(jogo2['odds']) and jogo2['odds'] > 0 else 1.0
            odd_total = odd1 * odd2
            
            # Calcular confiança média
            confianca_media = (jogo1['confianca'] + jogo2['confianca']) / 2
            
            # *** NOVO SISTEMA DE SCORE - PRIORIZANDO CONFIANÇA ALTA ***
            # Peso maior para confiança (70%) e menor para odds (30%)
            peso_confianca = 0.7
            peso_odds = 0.3
            
            # Score baseado principalmente na confiança
            score_confianca = confianca_media * 100  # Converte para escala 0-100
            
            # Bonus por odds (limitado para não dominar)
            bonus_odds = min((odd_total - 1) * 10, 20)  # Máximo 20 pontos bonus
            
            # Penalidade por odds muito altas (pouco prováveis)
            if odd_total > 5.0:
                bonus_odds *= 0.5  # Reduz bonus pela metade
            
            # *** BONUS POR PROXIMIDADE TEMPORAL ***
            bonus_proximidade = 0
            if jogo1['data_hora'] and jogo2['data_hora']:
                # Se as datas/horas forem iguais ou muito próximas
                if jogo1['data_hora'] == jogo2['data_hora']:
                    bonus_proximidade = 15  # Bônus por jogos no mesmo horário
                # Aqui você pode adicionar mais lógica de proximidade temporal
            
            # Score final - CONFIANÇA TEM PESO MAIOR
            score_final = (score_confianca * peso_confianca) + (bonus_odds * peso_odds) + bonus_proximidade
            
            multiplas_com_metricas.append({
                'jogos': jogos_ordenados,
                'odd_total': odd_total,
                'confianca_media': confianca_media,
                'score': score_final,
                'bonus_proximidade': bonus_proximidade
            })
        
        # *** ORDENAR PRINCIPALMENTE POR CONFIANÇA ***
        multiplas_com_metricas.sort(key=lambda x: (x['confianca_media'], x['score']), reverse=True)
        
        # Remover duplicatas finais
        multiplas_unicas = []
        combinacoes_vistas = set()
        
        for multipla in multiplas_com_metricas:
            chave = tuple(jogo['id'] for jogo in multipla['jogos'])
            
            if chave not in combinacoes_vistas:
                combinacoes_vistas.add(chave)
                multiplas_unicas.append(multipla)
        
        # Exibir estatísticas
        confianca_minima = min(m['confianca_media'] for m in multiplas_unicas[:10])
        confianca_maxima = max(m['confianca_media'] for m in multiplas_unicas[:10])
        
        print(f"🎯 {len(multiplas_unicas)} múltiplas únicas geradas")
        print(f"📊 Confiança das top 10: {confianca_minima:.1%} - {confianca_maxima:.1%}")
        
        # Exibir apenas as melhores por confiança
        multiplas_para_exibir = multiplas_unicas[:num_multiplas]
        
        for i, multipla in enumerate(multiplas_para_exibir, 1):
            print(f"\n🔮 MÚLTIPLA {i}:")
            print(f"   Odd Total: {multipla['odd_total']:.2f}")
            print(f"   Confiança Média: {multipla['confianca_media']:.1%} ⭐")  # Destaque na confiança
            if multipla['bonus_proximidade'] > 0:
                print(f"   🕒 Bônus Proximidade: +{multipla['bonus_proximidade']} pontos")
            
            for jogo in multipla['jogos']:
                odd_str = f"@{jogo['odds']:.2f}" if pd.notna(jogo['odds']) and jogo['odds'] > 0 else "@nan"
                confianca_individual = f"({jogo['confianca']:.1%})"
                print(f"   ✅ {jogo['time']} - {jogo['mercado']} {odd_str} {confianca_individual}")
            print("-" * 40)
        
        # Salvar todas as múltiplas
        self._salvar_multiplas_completas(multiplas_unicas)
        
        return multiplas_unicas
    def _salvar_multiplas_completas(self, multiplas):
        """Salvar todas as múltiplas em um arquivo CSV"""
        import csv
        
        with open('multiplas_completas.csv', 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['Multipla', 'Time_1', 'Mercado_1', 'Odd_1', 'Time_2', 'Mercado_2', 'Odd_2', 'Odd_Total', 'Confianca_Media'])
            
            for i, multipla in enumerate(multiplas, 1):
                jogo1, jogo2 = multipla['jogos']
                writer.writerow([
                    f"Multipla {i}",
                    jogo1['time'], jogo1['mercado'], jogo1['odds'],
                    jogo2['time'], jogo2['mercado'], jogo2['odds'],
                    multipla['odd_total'],
                    multipla['confianca_media']
                ])
        
        print(f"💾 Todas as {len(multiplas)} múltiplas salvas em: multiplas_completas.csv")
        
    # EXECUÇÃO PRINCIPAL INTELIGENTE
if __name__ == "__main__":
    print("🤖 SISTEMA DE ANÁLISE EVOLUTIVA DE APOSTAS")
    print("="*50)
    
    # CONFIGURAÇÃO - AJUSTE ESTES CAMINHOS
    config = {
        'base_treino': 'todos_ate_05-10-25.csv',      # Base completa COM históricos
        'base_futuros': 'jogos_futuros.csv',          # Apenas jogos futuros
        'modelo_salvo': 'modelo_apostas_evolutivo.joblib'
    }
    
    # Inicializar analisador
    analisador = AnalisadorApostasEvolutivo(
        base_treino_path=config['base_treino'],
        base_futuros_path=config['base_futuros'],
        modelo_path=config['modelo_salvo']
    )
    
    # OPÇÃO 1: Treinar/Atualizar modelo
    print("\n1. 🎯 Treinando modelo com dados atualizados...")
    sucesso_treino = analisador.treinar_modelo_evolutivo()
    
    if sucesso_treino:
        # OPÇÃO 2: Gerar previsões
        print("\n2. 🔮 Gerando previsões inteligentes...")
        previsoes = analisador.gerar_previsoes_futuras('previsoes_evolutivas.csv')
        
        print("\n" + "="*50)
        print("✅ PROCESSO CONCLUÍDO!")
        print("="*50)
        print("📁 Arquivos gerados:")
        print("   - modelo_apostas_evolutivo.joblib (modelo treinado)")
        print("   - previsoes_evolutivas.csv (previsões + múltiplas)")
        print("\n🎯 Próximos passos:")
        print("   - Adicione novos resultados à base de treino")
        print("   - Execute novamente para modelo mais inteligente")
        print("   - Use as múltiplas recomendadas para apostas")
    else:
        print("❌ Falha no processo. Verifique os arquivos de dados.")