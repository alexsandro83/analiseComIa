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
        """Gerar múltiplas de 2, 3 ou 4 times APENAS com confiança > 75%"""
        print("\n🎲 GERANDO MÚLTIPLAS DE ALTA CONFIANÇA (>75%):")
        print("="*50)
        
        if df_futuros is None or len(df_futuros) == 0:
            print("❌ Dados futuros não disponíveis para gerar múltiplas")
            return []
        
        jogos_excelentes = df_futuros[df_futuros['Recomendacao'] == 'EXCELENTE']
        jogos_bons = df_futuros[df_futuros['Recomendacao'] == 'BOA']
        
        # Combinar e ordenar por confiança
        todos_jogos = pd.concat([jogos_excelentes, jogos_bons])
        todos_jogos = todos_jogos.sort_values('Probabilidade_Sucesso', ascending=False)
        
        print(f"📊 Jogos excelentes: {len(jogos_excelentes)}, Jogos bons: {len(jogos_bons)}")
        
        if len(todos_jogos) < 2:
            print("❌ Número insuficiente de jogos para gerar múltiplas")
            return []
        
        # Criar lista de jogos com ID da partida
        jogos_lista = []
        
        for idx, row in todos_jogos.iterrows():
            time = row.get('Time', 'TIME_DESCONHECIDO')
            mercado = row.get('Tipo_Estatistica', 'MERCADO_DESCONHECIDO')
            
            # Extrair ID único da partida do Next Match
            id_partida = self._extrair_id_partida(row)
            
            jogos_lista.append({
                'id': f"{time}_{mercado}",
                'time': time,
                'mercado': mercado,
                'odds': row.get('Odds', 1.0),
                'confianca': row.get('Probabilidade_Sucesso', 0),
                'analise': row.get('Analise_Detalhada', ''),
                'id_partida': id_partida  # ID único da partida
            })
        
        # Remover duplicatas
        jogos_unicos = []
        ids_vistos = set()
        for jogo in jogos_lista:
            if jogo['id'] not in ids_vistos:
                ids_vistos.add(jogo['id'])
                jogos_unicos.append(jogo)
        
        print(f"🎯 Jogos únicos disponíveis: {len(jogos_unicos)}")
        
        # 🔥 NOVA LÓGICA: IDENTIFICAR JOGOS COM CONFLITO MAS CRIAR MÚLTIPLAS SEPARADAS
        jogos_para_multiplas = []
        grupos_conflito = {}
        
        # Identificar grupos de conflito (times do mesmo jogo)
        for jogo in jogos_unicos:
            if jogo['id_partida'] != 'DESCONHECIDO':
                if jogo['id_partida'] not in grupos_conflito:
                    grupos_conflito[jogo['id_partida']] = []
                grupos_conflito[jogo['id_partida']].append(jogo)
        
        print("🔍 Grupos de conflito identificados:")
        for partida_id, jogos in grupos_conflito.items():
            if len(jogos) > 1:
                times = [f"{j['time']}({j['confianca']:.1%})" for j in jogos]
                print(f"   🎯 {partida_id}: {', '.join(times)}")
        
        # MANTER TODOS OS JOGOS, mas marcar os que têm conflito
        for jogo in jogos_unicos:
            jogo['tem_conflito'] = False
            if jogo['id_partida'] in grupos_conflito and len(grupos_conflito[jogo['id_partida']]) > 1:
                jogo['tem_conflito'] = True
            jogos_para_multiplas.append(jogo)
        
        print(f"🎯 Jogos para múltiplas: {len(jogos_para_multiplas)} (incluindo opções de conflito)")
        
        if len(jogos_para_multiplas) < 2:
            print("❌ Número insuficiente de jogos para gerar múltiplas")
            return []
        
        # GERAR MÚLTIPLAS EVITANDO CONFLITOS DIRETOS
        todas_multiplas = []
        from itertools import combinations
        
        # 1. Múltiplas de 2 jogos
        print("🔄 Gerando múltiplas de 2 jogos (evitando conflitos diretos)...")
        comb_2_jogos = list(combinations(jogos_para_multiplas, 2))
        
        multiplas_validas_2 = 0
        for combo in comb_2_jogos:
            jogo1, jogo2 = combo
            
            # EVITAR: dois times do mesmo jogo na mesma múltipla
            if jogo1['id_partida'] == jogo2['id_partida'] and jogo1['id_partida'] != 'DESCONHECIDO':
                continue
            
            multipla = self._calcular_metricas_multipla([jogo1, jogo2])
            if multipla['confianca_media'] > 0.75:
                todas_multiplas.append(multipla)
                multiplas_validas_2 += 1
        
        print(f"   ✅ Múltiplas de 2 válidas: {multiplas_validas_2}")
        
        # 2. Múltiplas de 3 jogos
        multiplas_validas_3 = 0
        if len(jogos_para_multiplas) >= 3:
            print("🔄 Gerando múltiplas de 3 jogos (evitando conflitos diretos)...")
            comb_3_jogos = list(combinations(jogos_para_multiplas, 3))
            
            for combo in comb_3_jogos:
                # Verificar se há times do mesmo jogo no combo
                tem_conflito = False
                partidas_no_combo = {}
                
                for jogo in combo:
                    if jogo['id_partida'] != 'DESCONHECIDO':
                        if jogo['id_partida'] not in partidas_no_combo:
                            partidas_no_combo[jogo['id_partida']] = []
                        partidas_no_combo[jogo['id_partida']].append(jogo['time'])
                
                # Se alguma partida tem mais de 1 time, há conflito
                for partida, times in partidas_no_combo.items():
                    if len(times) > 1:
                        tem_conflito = True
                        break
                
                if tem_conflito:
                    continue
                
                multipla = self._calcular_metricas_multipla(combo)
                if multipla['confianca_media'] > 0.75:
                    todas_multiplas.append(multipla)
                    multiplas_validas_3 += 1
            
            print(f"   ✅ Múltiplas de 3 válidas: {multiplas_validas_3}")
        
        # 3. Múltiplas de 4 jogos
        multiplas_validas_4 = 0
        if len(jogos_para_multiplas) >= 4:
            print("🔄 Gerando múltiplas de 4 jogos (evitando conflitos diretos)...")
            comb_4_jogos = list(combinations(jogos_para_multiplas, 4))
            
            for combo in comb_4_jogos:
                # Verificar se há times do mesmo jogo no combo
                tem_conflito = False
                partidas_no_combo = {}
                
                for jogo in combo:
                    if jogo['id_partida'] != 'DESCONHECIDO':
                        if jogo['id_partida'] not in partidas_no_combo:
                            partidas_no_combo[jogo['id_partida']] = []
                        partidas_no_combo[jogo['id_partida']].append(jogo['time'])
                
                # Se alguma partida tem mais de 1 time, há conflito
                for partida, times in partidas_no_combo.items():
                    if len(times) > 1:
                        tem_conflito = True
                        break
                
                if tem_conflito:
                    continue
                
                multipla = self._calcular_metricas_multipla(combo)
                if multipla['confianca_media'] > 0.75:
                    todas_multiplas.append(multipla)
                    multiplas_validas_4 += 1
            
            print(f"   ✅ Múltiplas de 4 válidas: {multiplas_validas_4}")
        
        print(f"📊 Múltiplas geradas com confiança > 75%: {len(todas_multiplas)}")
        
        if len(todas_multiplas) == 0:
            print("❌ Nenhuma múltipla atingiu o mínimo de 75% de confiança")
            return []
        
        # ORDENAR POR CONFIANÇA (melhores primeiro)
        todas_multiplas.sort(key=lambda x: (x['confianca_media'], x['score']), reverse=True)
        
        # Remover duplicatas
        multiplas_unicas = []
        combinacoes_vistas = set()
        
        for multipla in todas_multiplas:
            chave = tuple(jogo['id'] for jogo in multipla['jogos'])
            if chave not in combinacoes_vistas:
                combinacoes_vistas.add(chave)
                multiplas_unicas.append(multipla)
        
        # CLASSIFICAR POR NÍVEL DE CONFIANÇA
        multiplas_altissima = [m for m in multiplas_unicas if m['confianca_media'] > 0.85]
        multiplas_alta = [m for m in multiplas_unicas if m['confianca_media'] > 0.75]
        
        print(f"\n🎯 CLASSIFICAÇÃO DAS MÚLTIPLAS (>75%):")
        print(f"   ⭐⭐⭐ ALTÍSSIMA (>85%): {len(multiplas_altissima)} múltiplas")
        print(f"   ⭐⭐ ALTA (>75%): {len(multiplas_alta)} múltiplas")
        print(f"   📊 Distribuição: 2-jogos={multiplas_validas_2}, 3-jogos={multiplas_validas_3}, 4-jogos={multiplas_validas_4}")
        
        # EXIBIR POR CATEGORIA
        self._exibir_multiplas_por_categoria(multiplas_altissima, "ALTÍSSIMA CONFIANÇA", "⭐⭐⭐", min(10, len(multiplas_altissima)))
        self._exibir_multiplas_por_categoria(multiplas_alta, "ALTA CONFIANÇA", "⭐⭐", min(10, len(multiplas_alta)))
        
        return multiplas_unicas
    def _tem_conflito_logico(self, jogo1, jogo2):
        """Verificar se há conflito lógico entre dois jogos - VERSÃO RIGOROSA"""
        # Se são da mesma partida, NÃO PERMITIR NENHUMA COMBINAÇÃO
        if jogo1['id_partida'] == jogo2['id_partida'] and jogo1['id_partida'] != 'DESCONHECIDO':
            print(f"❌ CONFLITO: {jogo1['time']} e {jogo2['time']} são do mesmo jogo {jogo1['id_partida']}")
            return True
        
        return False
        
    def _extrair_id_partida(self, row):
        """Extrair ID único da partida usando Stat + Next Match"""
        stat = str(row.get('Stat', ''))
        next_match = str(row.get('Next Match', ''))
        
        # Extrair time principal da coluna Stat (ex: "Brann have won their last 5 matches")
        time_principal = stat.split(' have ')[0] if ' have ' in stat else 'TIME_DESCONHECIDO'
        
        # Extrair time adversário e local da coluna Next Match
        if ' vs ' in next_match.lower():
            # Formato: "Home vs Atlante" ou "Away vs Atlante"
            partes = next_match.lower().split(' vs ')
            if len(partes) == 2:
                local = partes[0].strip()  # "home" ou "away"
                adversario = partes[1].strip().title()
                
                # Determinar times baseado no local
                if local == 'home':
                    time_casa = time_principal
                    time_visitante = adversario
                elif local == 'away':
                    time_casa = adversario
                    time_visitante = time_principal
                else:
                    return 'DESCONHECIDO'
                
                # Criar ID único ordenando os times
                times_ordenados = sorted([time_casa, time_visitante])
                id_partida = f"{times_ordenados[0]}_vs_{times_ordenados[1]}"
                
                print(f"🔍 DEBUG Partida: {time_principal} ({local}) vs {adversario} → ID: {id_partida}")
                return id_partida
        
        return 'DESCONHECIDO'
    def _calcular_metricas_multipla(self, combinacao_jogos):
        """Calcular métricas para uma múltipla de qualquer tamanho"""
        jogos_ordenados = sorted(combinacao_jogos, key=lambda x: x['id'])
        
        # Calcular odd total
        odd_total = 1.0
        confiancas = []
        
        for jogo in combinacao_jogos:
            odd = jogo['odds'] if pd.notna(jogo['odds']) and jogo['odds'] > 0 else 1.0
            odd_total *= odd
            confiancas.append(jogo['confianca'])
        
        # Calcular confiança média
        confianca_media = sum(confiancas) / len(confiancas)
        
        # Score com peso maior na confiança
        peso_confianca = 0.8
        peso_odds = 0.2
        
        score_confianca = confianca_media * 100
        bonus_odds = min((odd_total - 1) * 5, 15)
        
        # Penalidade por muitas seleções
        penalidade_tamanho = (len(combinacao_jogos) - 2) * 5
        
        score_final = (score_confianca * peso_confianca) + (bonus_odds * peso_odds) - penalidade_tamanho
        
        return {
            'jogos': jogos_ordenados,
            'odd_total': odd_total,
            'confianca_media': confianca_media,
            'score': score_final,
            'tamanho': len(combinacao_jogos)
        }

    def _exibir_multiplas_por_categoria(self, multiplas, categoria, icone, max_exibir):
        """Exibir múltiplas por categoria de confiança"""
        if not multiplas:
            return
        
        print(f"\n{icone} {categoria} {icone}")
        print("=" * 50)
        
        for i, multipla in enumerate(multiplas[:max_exibir], 1):
            tamanho_str = {
                2: "DUPLA",
                3: "TRIPLA", 
                4: "QUÁDRUPLA"
            }.get(multipla['tamanho'], f"{multipla['tamanho']} JOGOS")
            
            print(f"\n🔮 {tamanho_str} {i}:")
            print(f"   Odd Total: {multipla['odd_total']:.2f}")
            print(f"   Confiança Média: {multipla['confianca_media']:.1%} ⭐")
            print(f"   Nº de Seleções: {multipla['tamanho']}")
            
            for jogo in multipla['jogos']:
                odd_str = f"@{jogo['odds']:.2f}" if pd.notna(jogo['odds']) and jogo['odds'] > 0 else "@nan"
                conf_individual = f"({jogo['confianca']:.1%})"
                print(f"   ✅ {jogo['time']} - {jogo['mercado']} {odd_str} {conf_individual}")
            print("-" * 40)
        
    # EXECUÇÃO PRINCIPAL INTELIGENTE
if __name__ == "__main__":
    print("🤖 SISTEMA DE ANÁLISE EVOLUTIVA DE APOSTAS")
    print("="*50)
    
    # CONFIGURAÇÃO - AJUSTE ESTES CAMINHOS
    config = {
        'base_treino': 'todos_ate_05-10-25.csv',      # Base completa COM históricos
        'base_futuros': 'adam choi_dados_20251012_221510.csv',          # Apenas jogos futuros # poder ser o arquivo jogos_futuros.csv
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