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
            # PRIMEIRO: Verificar o formato real do arquivo
            print(f"📁 Tentando carregar: {csv_path}")
            
            # Lê as primeiras linhas para debug
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                primeira_linha = f.readline().strip()
                segunda_linha = f.readline().strip()
                print(f"🔍 Primeira linha: {primeira_linha}")
                print(f"🔍 Segunda linha: {segunda_linha}")
            
            # Tenta diferentes delimitadores
            delimitadores = [';', ',', '\t', '|']
            
            for delim in delimitadores:
                try:
                    print(f"🔧 Tentando delimitador: '{delim}'")
                    df = pd.read_csv(csv_path, delimiter=delim, encoding='utf-8-sig')
                    print(f"✅ Sucesso com delimitador '{delim}': {len(df)} linhas, {len(df.columns)} colunas")
                    print(f"📋 Colunas: {list(df.columns)}")
                    
                    if len(df.columns) > 1:
                        return df
                except Exception as e:
                    print(f"❌ Falha com delimitador '{delim}': {e}")
                    continue
            
            # Se nenhum delimitador funcionou, tentar carregamento automático
            print("🔄 Tentando carregamento automático...")
            df = pd.read_csv(csv_path, encoding='utf-8-sig', engine='python')
            print(f"📊 Carregamento automático: {len(df)} linhas, {len(df.columns)} colunas")
            
            # Se ainda estiver como uma coluna, tentar split manual
            if len(df.columns) == 1:
                print("🔄 Dividindo coluna única manualmente...")
                coluna_unica = df.columns[0]
                # Divide pela vírgula (que parece ser o seu delimitador)
                dados_divididos = df[coluna_unica].str.split(',', expand=True)
                
                # Pega o cabeçalho da primeira linha
                cabecalho = dados_divididos.iloc[0].str.strip()
                dados_divididos = dados_divididos[1:]  # Remove a linha do cabeçalho
                dados_divididos.columns = cabecalho
                
                print(f"✅ Divisão manual: {len(dados_divididos)} linhas, {len(dados_divididos.columns)} colunas")
                return dados_divididos
                
            return df
            
        except pd.errors.ParserError as e:
            print(f"❌ Erro de parsing detectado: {e}")
            print("🔧 Tentando corrigir automaticamente...")
            
            # Lê o arquivo linha por linha para identificar o problema
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                linhas = f.readlines()
            
            # Identifica a linha problemática
            for i, linha in enumerate(linhas, 1):
                if len(linha.split(',')) != 7:  # Espera 7 colunas
                    print(f"📝 Linha {i} problemática: {linha.strip()}")
            
            # Tenta carregar com tratamento de erro mais flexível
            try:
                df = pd.read_csv(csv_path, delimiter=',', encoding='utf-8-sig', 
                            on_bad_lines='skip',  # Pula linhas problemáticas
                            engine='python')
                print(f"✅ Carregado com {len(df)} linhas após correção automática")
                return df
            except:
                # Última tentativa: carrega manualmente
                print("🔄 Carregando manualmente...")
                dados_corrigidos = []
                cabecalho = linhas[0].strip().split(',')
                
                for i, linha in enumerate(linhas[1:], 2):  # Começa da linha 2 (pula cabeçalho)
                    campos = linha.strip().split(',')
                    if len(campos) == len(cabecalho):
                        dados_corrigidos.append(campos)
                    else:
                        print(f"⚠️  Linha {i} ignorada: número de campos incorreto ({len(campos)} vs {len(cabecalho)})")
                
                df = pd.DataFrame(dados_corrigidos, columns=cabecalho)
                print(f"✅ Carregamento manual: {len(df)} linhas processadas")
                return df
                
        except UnicodeDecodeError:
            try:
                return pd.read_csv(csv_path, delimiter=',', encoding='latin-1')
            except:
                return pd.read_csv(csv_path, delimiter=',', encoding='iso-8859-1')
        except Exception as e:
            print(f"❌ Erro crítico ao carregar dados: {e}")
            return None
    
    def treinar_modelo_evolutivo(self, forcar_retreino=False):
        """Treinar ou atualizar modelo com dados mais recentes"""
        print("🔄 INICIANDO TREINAMENTO EVOLUTIVO...")
        
        # Verificar se modelo existe e se deve atualizar
        if os.path.exists(self.modelo_path) and not forcar_retreino:
            print("📁 Modelo existente encontrado. Verificando necessidade de atualização...")
            if not self._verificar_necessidade_atualizacao():
                print("✅ Modelo atualizado. Carregando modelo existente...")
                if self.carregar_modelo():
                    return True
        
        if self.base_treino_path is None:
            print("❌ Caminho da base de treino não especificado")
            return False
            
        # Carregar e preparar dados
        print("📥 Carregando dados de treino...")
        self.df_treino = self.carregar_dados(self.base_treino_path)
        
        if self.df_treino is None or len(self.df_treino) == 0:
            print("❌ Falha ao carregar dados de treino")
            return False
        
        print(f"📊 Dados carregados: {len(self.df_treino)} registros")
        print(f"📋 Colunas: {list(self.df_treino.columns)}")
        
        self._preparar_dados_treino()
        
        if not hasattr(self, 'df_treino_limpo') or len(self.df_treino_limpo) < 10:
            print("❌ Dados insuficientes para treinamento")
            return False
        
        # Treinar novo modelo
        print("🎯 Treinando novo modelo com dados atualizados...")
        try:
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
            
        except Exception as e:
            print(f"❌ Erro no treinamento: {e}")
            return False

    def _verificar_necessidade_atualizacao(self):
        """Verificar se base de treino tem novos dados"""
        if not os.path.exists(self.modelo_path) or self.base_treino_path is None:
            return True
            
        # Carregar dados atuais
        df_atual = self.carregar_dados(self.base_treino_path)
        
        if df_atual is None or len(df_atual) == 0:
            print("❌ Falha ao carregar dados para verificação")
            return True
        
        # CORREÇÃO: Verificar se a coluna existe após carregamento correto
        if 'Situacao' not in df_atual.columns:
            print("❌ Coluna 'Situacao' não encontrada após carregamento")
            print(f"📋 Colunas disponíveis: {list(df_atual.columns)}")
            return True
        
        # Filtrar dados válidos
        df_atual_limpo = df_atual[df_atual['Situacao'].notna() & 
                                (df_atual['Situacao'] != '')]
        
        # Carregar info do modelo salvo
        try:
            modelo_data = joblib.load(self.modelo_path)
            amostras_anteriores = modelo_data.get('amostras_treino', 0)
            
            print(f"📊 Comparação: Modelo atual {amostras_anteriores} vs Dados {len(df_atual_limpo)}")
            
            # Se tem pelo menos 10% mais dados, retreinar
            if len(df_atual_limpo) > amostras_anteriores * 1.1:
                print(f"📈 Novos dados detectados: {amostras_anteriores} → {len(df_atual_limpo)}")
                return True
            else:
                print(f"✅ Modelo já está atualizado")
                
        except Exception as e:
            print(f"⚠️  Erro ao verificar modelo: {e}")
            return True
            
        return False

    def _preparar_dados_treino(self):
        """Preparar dados para treino"""
        print("📊 Preparando dados para treino...")

        # DEBUG: Mostrar informações dos dados
        print(f"📋 Colunas disponíveis: {list(self.df_treino.columns)}")
        print(f"🔍 Primeiras linhas:")
        print(self.df_treino.head(2))
        
        def corrigir_data(date_str):
            try:
                # Sua base já tem data no formato correto: "2025-09-22 21:15:00"
                return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S', errors='coerce')
            except:
                try:
                    # Tentativa alternativa
                    return pd.to_datetime(date_str, errors='coerce')
                except:
                    return pd.NaT
        
        # Processar Date
        if 'Date' in self.df_treino.columns:
            self.df_treino['Date'] = self.df_treino['Date'].apply(corrigir_data)
            print(f"✅ Datas processadas: {self.df_treino['Date'].notna().sum()} válidas")
        else:
            print("⚠️  Coluna 'Date' não encontrada")
            self.df_treino['Date'] = pd.NaT
        
        # Processar Odds
        if 'Odds' in self.df_treino.columns:
            self.df_treino['Odds'] = pd.to_numeric(self.df_treino['Odds'], errors='coerce')
            print(f"✅ Odds processadas: {self.df_treino['Odds'].notna().sum()} válidas")
        else:
            print("❌ Coluna 'Odds' não encontrada - ESSENCIAL")
            return
        
        # CORREÇÃO: Usar coluna 'Situacao' que existe na sua base
        if 'Situacao' in self.df_treino.columns:
            situacao_map = {'VERDADEIRO': 1, 'Verdadeiro': 1, 'FALSO': 0, 'Falso': 0, 'GREEN': 1, 'RED': 0, 'WIN': 1, 'LOSS': 0}
            self.df_treino['Target'] = self.df_treino['Situacao'].map(situacao_map).fillna(-1)
            print(f"✅ Target criado: {len(self.df_treino[self.df_treino['Target'] != -1])} amostras válidas")
        else:
            print("❌ Coluna 'Situacao' não encontrada - ESSENCIAL")
            return
        
        # Features - verificar colunas essenciais
        if 'Stat' in self.df_treino.columns:
            self.df_treino['Tipo_Estatistica'] = self.df_treino['Stat'].apply(self._classificar_estatistica)
            self.df_treino['Tamanho_Streak'] = self.df_treino['Stat'].apply(self._extrair_streak)
            print(f"✅ Stats processados: {len(self.df_treino)} registros")
        else:
            print("❌ Coluna 'Stat' não encontrada - ESSENCIAL")
            return
        
        if 'Next Match' in self.df_treino.columns:
            self.df_treino['Local_Jogo'] = self.df_treino['Next Match'].apply(self._extrair_local)
            print("✅ Local do jogo processado")
        else:
            print("⚠️  Coluna 'Next Match' não encontrada")
            self.df_treino['Local_Jogo'] = 'NEUTRO'
        
        if 'League' in self.df_treino.columns:
            self.df_treino['Liga_Categoria'] = self.df_treino['League'].apply(self._classificar_liga)
            print("✅ Liga categorizada")
        else:
            print("⚠️  Coluna 'League' não encontrada")
            self.df_treino['Liga_Categoria'] = 'MEDIA_CONFIABILIDADE'

        # Filtrar apenas dados com target válido
        self.df_treino_limpo = self.df_treino[self.df_treino['Target'] != -1].copy()
        print(f"🎯 Dados finais para treino: {len(self.df_treino_limpo)} registros válidos")
        
        # Mostrar distribuição
        if len(self.df_treino_limpo) > 0:
            verd_count = (self.df_treino_limpo['Target'] == 1).sum()
            fals_count = (self.df_treino_limpo['Target'] == 0).sum()
            print(f"📊 Distribuição: VERDADEIRO={verd_count}, FALSO={fals_count}")
    
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
        print("📊 Preparando dados futuros...")
        
        # Processar Date - converter do formato inglês
        if 'Date' in df_futuros.columns:
            df_futuros['Date'] = df_futuros['Date'].apply(self._converter_data_ingles_para_brasil)
            print(f"✅ Datas processadas: {df_futuros['Date'].notna().sum()} válidas")
        else:
            print("⚠️  Coluna 'Date' não encontrada")
            df_futuros['Date'] = "DATA_INDISPONIVEL"
        
        # Processar Odds - converter para numérico
        if 'Odds' in df_futuros.columns:
            df_futuros['Odds'] = pd.to_numeric(df_futuros['Odds'], errors='coerce')
            print(f"✅ Odds processadas: {df_futuros['Odds'].notna().sum()} válidas")
        else:
            print("⚠️  Coluna 'Odds' não encontrada")
            df_futuros['Odds'] = np.nan
        
        # Processar Stat - extrair informações
        if 'Stat' in df_futuros.columns:
            df_futuros['Tipo_Estatistica'] = df_futuros['Stat'].apply(self._classificar_estatistica)
            df_futuros['Tamanho_Streak'] = df_futuros['Stat'].apply(self._extrair_streak)
            df_futuros['Time'] = df_futuros['Stat'].apply(lambda x: x.split(' have ')[0] if ' have ' in str(x) else 'TIME_DESCONHECIDO')
            print(f"✅ Stats processados: {len(df_futuros)} registros")
        else:
            print("❌ Coluna 'Stat' não encontrada - ESSENCIAL")
            df_futuros['Tipo_Estatistica'] = 'OUTRO'
            df_futuros['Tamanho_Streak'] = 1
            df_futuros['Time'] = 'TIME_DESCONHECIDO'
        
        # Processar Next Match - extrair local e formatar
        if 'Next Match' in df_futuros.columns:
            df_futuros['Local_Jogo'] = df_futuros['Next Match'].apply(self._extrair_local)
            # Aplicar formatação diretamente na coluna Next Match existente
            df_futuros['Next Match'] = df_futuros.apply(self._formatar_next_match, axis=1)
            print("✅ Next Match processado e formatado")
        else:
            print("⚠️  Coluna 'Next Match' não encontrada")
            df_futuros['Local_Jogo'] = 'NEUTRO'
        
        # Processar League - classificar categoria
        if 'League' in df_futuros.columns:
            df_futuros['Liga_Categoria'] = df_futuros['League'].apply(self._classificar_liga)
            print("✅ Liga categorizada")
        else:
            print("⚠️  Coluna 'League' não encontrada")
            df_futuros['Liga_Categoria'] = 'MEDIA_CONFIABILIDADE'
        
        # Adicionar coluna Situacao vazia para consistência
        if 'Situacao' not in df_futuros.columns:
            df_futuros['Situacao'] = ''
            print("✅ Coluna Situacao adicionada (vazia para jogos futuros)")
        
        print(f"📊 Dados futuros preparados: {len(df_futuros)} registros")
        print(f"📋 Colunas disponíveis: {list(df_futuros.columns)}")
        
        return df_futuros
    def _formatar_next_match(self, row):
        """Formatar Next Match para mostrar time principal em maiúsculas"""
        next_match = str(row.get('Next Match', ''))
        time_principal = str(row.get('Time', ''))
        
        if not time_principal or time_principal == 'TIME_DESCONHECIDO':
            return next_match.upper()
        
        # Substituir "home" ou "away" pelo nome do time em maiúsculas
        next_match_lower = next_match.lower()
        if 'home' in next_match_lower:
            resultado = next_match_lower.replace('home', time_principal.upper())
            print(f"🏠 Next Match formatado: '{next_match}' -> '{resultado}'")
            return resultado
        elif 'away' in next_match_lower:
            resultado = next_match_lower.replace('away', time_principal.upper())
            print(f"✈️  Next Match formatado: '{next_match}' -> '{resultado}'")
            return resultado
        else:
            return next_match.upper()
        

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
    def _formatar_next_match(self, row):
        """Formatar Next Match para mostrar time principal em maiúsculas"""
        next_match = str(row.get('Next Match', ''))
        time_principal = str(row.get('Time', ''))
        
        if not time_principal or time_principal == 'TIME_DESCONHECIDO':
            return next_match.upper()
        
        # Substituir "home" ou "away" pelo nome do time em maiúsculas
        if 'home' in next_match.lower():
            return next_match.lower().replace('home', time_principal.upper())
        elif 'away' in next_match.lower():
            return next_match.lower().replace('away', time_principal.upper())
        else:
            return next_match.upper()
    
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
    
    def gerar_previsoes_futuras(self, output_path='previsoes_evolutivas.csv'):
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
        
        # Adicionar previsões ao DataFrame - APENAS AS COLUNAS EXISTENTES
        colunas_previsao = ['Probabilidade_Sucesso', 'Previsao', 'Padrao', 'Recomendacao', 'Analise_Detalhada', 'Bonus_Total']
        for col in colunas_previsao:
            df_futuros[col] = [p[col] for p in previsoes_detalhadas]
        
        # Formatar Date no formato brasileiro
        if 'Date' in df_futuros.columns:
            df_futuros['Date'] = df_futuros['Date'].apply(self._formatar_data_saida)
        
        # Ordenar por probabilidade e recomendação
        df_futuros['Score_Prioridade'] = df_futuros['Probabilidade_Sucesso'] * df_futuros['Odds']
        df_futuros = df_futuros.sort_values(['Recomendacao', 'Score_Prioridade'], ascending=[False, False])
        
        # REMOVER COLUNAS TEMPORÁRIAS QUE NÃO DEVEM APARECER NO CSV FINAL
        colunas_para_manter = [
            'League', 'Stat', 'Next Match', 'Odds', 'Date', 'Situação',
            'Tipo_Estatistica', 'Liga_Categoria',
            'Probabilidade_Sucesso', 'Previsao', 'Padrao', 'Recomendacao', 'Analise_Detalhada', 
        ]
        # ficam de fora
        # 'Tamanho_Streak',  'Time', 'Local_Jogo',         'Bonus_Total'	'Score_Prioridade', 



        # Manter apenas as colunas que existem no DataFrame
        colunas_existentes = [col for col in colunas_para_manter if col in df_futuros.columns]
        df_futuros = df_futuros[colunas_existentes]
        
        # Gerar múltiplas recomendadas
        self._gerar_multiplas_recomendadas(df_futuros)
        
        # Salvar resultados
        df_futuros.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')
        print(f"✅ Previsões salvas em: {output_path}")
        print(f"📊 Total de jogos analisados: {len(df_futuros)}")
        print(f"🎯 Jogos EXCELENTES: {len(df_futuros[df_futuros['Recomendacao'] == 'EXCELENTE'])}")
        print(f"👍 Jogos BONS: {len(df_futuros[df_futuros['Recomendacao'] == 'BOA'])}")
        
        return df_futuros
        
    def _converter_data_ingles_para_brasil(self, date_str):
        
        """Converter data do formato 'Sunday, 12 October 12:00' para '12/10/2025 12:00'"""
        if pd.isna(date_str) or date_str == '' or date_str == 'DATA_INDISPONIVEL':
            return "DATA_INDISPONIVEL"
        
        try:
            date_str_clean = str(date_str).strip()
            
            # Mapeamento dos meses em inglês para números
            meses_ingles = {
                'january': '01', 'february': '02', 'march': '03', 'april': '04',
                'may': '05', 'june': '06', 'july': '07', 'august': '08',
                'september': '09', 'october': '10', 'november': '11', 'december': '12'
            }
            
            # Remove o dia da semana e espaços extras
            date_clean = date_str_clean.replace(',', '').strip()
            
            # Divide a string: "Sunday 12 October 12:00"
            partes = date_clean.split()
            
            if len(partes) >= 4:
                # Pega dia, mês e hora
                dia = partes[1].zfill(2)  # "12" -> "12"
                mes_ingles = partes[2].lower()  # "October" -> "october"
                hora = partes[3]  # "12:00"
                
                # Converte mês inglês para número
                mes_numero = meses_ingles.get(mes_ingles, '01')
                
                # Assume ano atual (ou próximo se estivermos no final do ano)
                ano_atual = datetime.now().year
                data_obj = datetime(ano_atual, int(mes_numero), int(dia))
                
                # Se a data já passou neste ano, usa próximo ano
                if data_obj < datetime.now():
                    data_obj = datetime(ano_atual + 1, int(mes_numero), int(dia))
                
                # Formata para "dd/mm/aaaa hh:mm"
                data_formatada = data_obj.strftime('%d/%m/%Y') + f' {hora}'
                print(f"📅 Data convertida: '{date_str}' -> '{data_formatada}'")
                return data_formatada
            else:
                print(f"⚠️  Formato de data não reconhecido: {date_str}")
                return "DATA_INDISPONIVEL"
                
        except Exception as e:
            print(f"❌ Erro ao converter data '{date_str}': {e}")
            return "DATA_INDISPONIVEL"

    def _formatar_data_saida(self, date_val):
        """Formatar data para saída no formato dd/mm/aaaa hh:mm"""
        if pd.isna(date_val) or date_val == '':
            return "DATA_INDISPONIVEL"
        
        try:
            # Se já for datetime, formatar diretamente
            if isinstance(date_val, (pd.Timestamp, datetime)):
                return date_val.strftime('%d/%m/%Y %H:%M')
            
            # Se for string, tentar converter do formato "Wednesday, 15 October 19:00"
            date_str = str(date_val).strip()
            
            # Verificar se está no formato inglês com dia da semana
            if ',' in date_str and any(day in date_str.lower() for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
                return self._converter_data_ingles_para_brasil(date_str)
            
            # Tentar outros formatos conhecidos
            for fmt in ['%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M', '%m/%d/%Y %H:%M', '%d-%m-%Y %H:%M']:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime('%d/%m/%Y %H:%M')
                except:
                    continue
            
            # Se não conseguir converter, retorna original
            return date_str
            
        except Exception as e:
            print(f"⚠️  Erro ao formatar data '{date_val}': {e}")
            return str(date_val)
    
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
    
    def _gerar_multiplas_recomendadas(self, df_futuros, num_multiplas=7):
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
        'base_futuros': 'adam choi_dados_20251017_001451.csv',          # Apenas jogos futuros # poder ser o arquivo jogos_futuros.csv
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