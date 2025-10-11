import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
from datetime import datetime

class AnalisadorApostas:
    def __init__(self, csv_path):
        # Tentar diferentes encodings
        try:
            self.df = pd.read_csv(csv_path, delimiter=';', encoding='utf-8-sig')
        except:
            try:
                self.df = pd.read_csv(csv_path, delimiter=';', encoding='latin-1')
            except:
                self.df = pd.read_csv(csv_path, delimiter=';', encoding='iso-8859-1')
        
        self.model = None
        self.encoder = LabelEncoder()
        self.jogos_complementares = {}
        
    def limpar_dados(self):
        """Limpar e preparar os dados para análise"""
        # Converter datas - CORRIGINDO O ANO
        def corrigir_data(date_str):
            try:
                # Converter para datetime
                dt = pd.to_datetime(date_str, format='%A, %d %B %H:%M', errors='coerce')
                if pd.isna(dt):
                    return dt
                # Corrigir o ano para 2025 (assumindo que os dados são de 2025)
                return dt.replace(year=2025)
            except:
                return pd.NaT
        
        self.df['Date'] = self.df['Date'].apply(corrigir_data)
        
        # Converter odds para numérico
        self.df['Odds'] = pd.to_numeric(self.df['Odds'], errors='coerce')
        
        # Criar target binário para jogos com histórico
        situacao_map = {
            'VERDADEIRO': 1, 'Verdadeiro': 1, 'VERDADEIR': 1,
            'FALSO': 0, 'Falso': 0, 'FALS': 0
        }
        
        if 'Situacao' in self.df.columns:
            self.df['Target'] = self.df['Situacao'].map(situacao_map).fillna(-1)
        else:
            self.df['Target'] = -1  # Todos são futuros
        
        # Extrair características das estatísticas
        self.df['Tipo_Estatistica'] = self.df['Stat'].apply(self._classificar_estatistica)
        self.df['Tamanho_Streak'] = self.df['Stat'].apply(self._extrair_streak)
        self.df['Local_Jogo'] = self.df['Next Match'].apply(self._extrair_local)
        self.df['Time'] = self.df['Stat'].apply(self._extrair_time)
        
        # Identificar jogos complementares
        self._identificar_jogos_complementares()
        
        # Dados para treino (apenas com target válido)
        self.df_limpo = self.df[self.df['Target'] != -1].copy()
        
    def _extrair_time(self, stat):
        """Extrair nome do time da estatística"""
        try:
            return stat.split(' have ')[0].strip()
        except:
            return 'TIME_DESCONHECIDO'
    
    def _identificar_jogos_complementares(self):
        """Identificar jogos que têm estatísticas de ambos os times"""
        jogos_unicos = {}
        
        for idx, row in self.df.iterrows():
            if pd.isna(row['Date']):
                continue
                
            chave_jogo = f"{row['Date']}_{row['League']}"
            
            if chave_jogo not in jogos_unicos:
                jogos_unicos[chave_jogo] = {
                    'league': row['League'],
                    'date': row['Date'],
                    'next_match': row['Next Match'],
                    'stats': []
                }
            
            jogos_unicos[chave_jogo]['stats'].append({
                'index': idx,
                'time': row['Time'],
                'stat': row['Stat'],
                'tipo': row['Tipo_Estatistica'],
                'odds': row['Odds'],
                'situacao': row['Situacao'] if 'Situacao' in row else None
            })
        
        # Filtrar apenas jogos com múltiplas estatísticas de times diferentes
        self.jogos_complementares = {}
        
        for chave, jogo in jogos_unicos.items():
            if len(jogo['stats']) >= 2:
                times = [s['time'] for s in jogo['stats']]
                # Verificar se são times diferentes
                if len(set(times)) >= 2:
                    self.jogos_complementares[chave] = jogo
        
        print(f"Jogos com estatisticas complementares identificados: {len(self.jogos_complementares)}")
    
    def _classificar_estatistica(self, stat):
        """Classificar o tipo de estatística"""
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
        """Extrair tamanho do streak da estatística"""
        matches = re.findall(r'last (\d+)', str(stat))
        return int(matches[0]) if matches else 1
    
    def _extrair_local(self, next_match):
        """Extrair se é jogo em casa ou fora"""
        match_str = str(next_match).lower()
        if 'home' in match_str:
            return 'CASA'
        elif 'away' in match_str:
            return 'FORA'
        else:
            return 'NEUTRO'
    
    def _classificar_padrao(self, probabilidade, bonus_complementar=0):
        """Classificar o padrão baseado na probabilidade"""
        prob_ajustada = probabilidade + bonus_complementar
        
        if prob_ajustada > 0.8:
            return 'PADRAO_FORTISSIMO'
        elif prob_ajustada > 0.7:
            return 'PADRAO_FORTE'
        elif prob_ajustada > 0.6:
            return 'PADRAO_SOLIDO'
        else:
            return 'PADRAO_REGULAR'
    
    def _classificar_recomendacao(self, probabilidade, bonus_complementar=0):
        """Classificar a recomendação"""
        prob_ajustada = probabilidade + bonus_complementar
        
        if prob_ajustada > 0.75:
            return 'EXCELENTE'
        elif prob_ajustada > 0.6:
            return 'BOA'
        else:
            return 'REGULAR'
    
    def _calcular_bonus_complementar(self, league, next_match, date):
        """Calcular bonus para jogos com estatísticas complementares"""
        chave_jogo = f"{date}_{league}"
        
        if chave_jogo in self.jogos_complementares:
            jogo = self.jogos_complementares[chave_jogo]
            
            # Verificar tipos de estatísticas presentes
            tipos_stats = [s['tipo'] for s in jogo['stats']]
            
            # Bônus baseado na combinação de estatísticas
            if 'VITORIA' in tipos_stats and 'DERROTA' in tipos_stats:
                return 0.15  # Bônus alto para vitória × derrota
            elif 'OVER_2.5' in tipos_stats and 'BTTS' in tipos_stats:
                return 0.12  # Bônus para Over 2.5 × BTTS
            elif len(set(tipos_stats)) >= 2:
                return 0.08  # Bônus para outras combinações
            
        return 0.0
    
    def analisar_jogos_complementares(self):
        """Analisar especificamente os jogos com estatísticas complementares"""
        if not self.jogos_complementares:
            print("Nenhum jogo complementar identificado")
            return
        
        resultados = []
        
        for chave, jogo in self.jogos_complementares.items():
            # Calcular taxa de acerto para este jogo
            situacoes = [s['situacao'] for s in jogo['stats'] if s['situacao'] in ['VERDADEIRO', 'Verdadeiro', 'FALSO', 'Falso']]
            
            if situacoes:
                acertos = sum(1 for s in situacoes if s in ['VERDADEIRO', 'Verdadeiro'])
                taxa_acerto = acertos / len(situacoes)
            else:
                taxa_acerto = None
            
            # Analisar combinação de estatísticas
            tipos_stats = [s['tipo'] for s in jogo['stats']]
            combinacao = " + ".join(sorted(set(tipos_stats)))
            
            resultados.append({
                'Jogo': chave,
                'Liga': jogo['league'],
                'Confronto': jogo['next_match'],
                'Combinacao_Stats': combinacao,
                'Total_Stats': len(jogo['stats']),
                'Taxa_Acerto': taxa_acerto,
                'Stats_Detalhadas': [f"{s['time']}: {s['tipo']} (Odds: {s['odds']})" for s in jogo['stats']]
            })
        
        # Criar DataFrame com resultados
        df_resultados = pd.DataFrame(resultados)
        
        # Ordenar por taxa de acerto (quando disponível)
        if not df_resultados.empty:
            print("\n" + "="*80)
            print("ANALISE DE JOGOS COM ESTATISTICAS COMPLEMENTARES")
            print("="*80)
            
            # Mostrar estatísticas gerais
            total_jogos = len(df_resultados)
            jogos_com_resultado = df_resultados[df_resultados['Taxa_Acerto'].notna()]
            
            if not jogos_com_resultado.empty:
                taxa_media = jogos_com_resultado['Taxa_Acerto'].mean()
                print(f"Taxa media de acerto em jogos complementares: {taxa_media:.1%}")
                print(f"Total de jogos analisados: {len(jogos_com_resultado)}")
            
            # Mostrar combinações mais comuns
            combinacoes = df_resultados['Combinacao_Stats'].value_counts()
            print(f"\nCOMBINACOES MAIS FREQUENTES:")
            for combinacao, count in combinacoes.head(5).items():
                print(f"  {combinacao}: {count} jogos")
            
            # Salvar análise detalhada
            df_resultados.to_csv('analise_jogos_complementares.csv', index=False, encoding='utf-8-sig')
            print(f"\nAnalise detalhada salva em: analise_jogos_complementares.csv")
            
            return df_resultados
    
    def treinar_modelo(self):
        """Treinar modelo de machine learning"""
        if len(self.df_limpo) < 10:
            print("Dados insuficientes para treinar modelo. Usando analise qualitativa.")
            self.model = None
            return
        
        # Preparar features
        features = ['Odds', 'Tamanho_Streak', 'Tipo_Estatistica', 'Local_Jogo']
        
        # Codificar variáveis categóricas
        X = self.df_limpo[features].copy()
        X['Tipo_Estatistica'] = self.encoder.fit_transform(X['Tipo_Estatistica'])
        X['Local_Jogo'] = LabelEncoder().fit_transform(X['Local_Jogo'])
        
        # Target
        y = self.df_limpo['Target']
        
        # Treinar modelo
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Avaliar
        accuracy = self.model.score(X_test, y_test)
        print(f"Acuracia do modelo: {accuracy:.2%}")
        
        # Mostrar importância das features
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nImportancia das variaveis:")
        print(feature_importance)
    
    def analisar_jogo(self, league, stat, next_match, odds, date):
        """Analisar um jogo específico"""
        if self.model is None:
            return self._analise_basica_futuro({
                'League': league, 'Stat': stat, 'Next Match': next_match,
                'Odds': odds, 'Date': date
            }, has_odds=not pd.isna(odds))
        
        # Preparar dados do jogo
        tipo_estatistica = self._classificar_estatistica(stat)
        tamanho_streak = self._extrair_streak(stat)
        local_jogo = self._extrair_local(next_match)
        
        # Calcular bônus para jogos complementares
        bonus_complementar = self._calcular_bonus_complementar(league, next_match, date)
        
        # Criar feature vector
        features = pd.DataFrame([{
            'Odds': odds,
            'Tamanho_Streak': tamanho_streak,
            'Tipo_Estatistica': tipo_estatistica,
            'Local_Jogo': local_jogo
        }])
        
        # Codificar
        features['Tipo_Estatistica'] = self.encoder.transform(features['Tipo_Estatistica'])
        features['Local_Jogo'] = LabelEncoder().fit_transform(features['Local_Jogo'])
        
        # Prever
        probabilidade = self.model.predict_proba(features)[0][1]
        previsao = self.model.predict(features)[0]
        
        # Aplicar bônus para jogos complementares
        probabilidade_ajustada = min(probabilidade + bonus_complementar, 0.95)
        
        # Classificar padrão e recomendação
        padrao = self._classificar_padrao(probabilidade, bonus_complementar)
        recomendacao = self._classificar_recomendacao(probabilidade, bonus_complementar)
        
        # Gerar análise textual
        analise = self._gerar_analise_texto(probabilidade_ajustada, previsao, tipo_estatistica, 
                                          tamanho_streak, odds, padrao, bonus_complementar, 
                                          is_previsao=True)
        
        return {
            'Probabilidade_Sucesso': probabilidade_ajustada,
            'Previsao': 'VERDADEIRO' if previsao == 1 else 'FALSO',
            'Padrao': padrao,
            'Recomendacao': recomendacao,
            'Analise_Texto': analise,
            'Bonus_Complementar': bonus_complementar
        }
    
    def _analise_basica_futuro(self, row, has_odds=True):
        """Análise especializada para jogos futuros"""
        tipo_estatistica = self._classificar_estatistica(row['Stat'])
        tamanho_streak = self._extrair_streak(row['Stat'])
        local_jogo = self._extrair_local(row['Next Match'])
        
        # Calcular bônus complementar mesmo sem odds
        bonus_complementar = self._calcular_bonus_complementar(
            row['League'], row['Next Match'], row['Date']
        )
        
        # Lógica de previsão baseada em padrões históricos
        if tipo_estatistica == 'VITORIA':
            if tamanho_streak >= 6:
                padrao = 'PADRAO_FORTE'
                recomendacao = 'EXCELENTE' if bonus_complementar > 0 else 'BOA'
                previsao = 'VERDADEIRO'
                prob = 0.75 + bonus_complementar
            else:
                padrao = 'PADRAO_SOLIDO'
                recomendacao = 'BOA'
                previsao = 'VERDADEIRO'
                prob = 0.65 + bonus_complementar
                
        elif tipo_estatistica == 'DERROTA':
            if tamanho_streak >= 6:
                padrao = 'PADRAO_FORTE'
                recomendacao = 'EXCELENTE' if bonus_complementar > 0 else 'BOA'
                previsao = 'VERDADEIRO'
                prob = 0.72 + bonus_complementar
            else:
                padrao = 'PADRAO_SOLIDO'
                recomendacao = 'BOA'
                previsao = 'VERDADEIRO'
                prob = 0.62 + bonus_complementar
                
        elif tipo_estatistica == 'OVER_2.5':
            padrao = 'PADRAO_SOLIDO'
            recomendacao = 'BOA'
            previsao = 'VERDADEIRO'
            prob = 0.60 + bonus_complementar
            
        else:  # BTTS, EMPATE, etc.
            padrao = 'PADRAO_REGULAR'
            recomendacao = 'REGULAR'
            previsao = 'VERDADEIRO'  # Assume que streak continua por padrão
            prob = 0.55 + bonus_complementar
        
        # Ajustar probabilidade máxima
        prob = min(prob, 0.90)
        
        # Gerar análise textual
        odds_text = f"(Odds: {row['Odds']})" if has_odds and not pd.isna(row['Odds']) else "(Odds: N/D)"
        
        analise = f"{padrao.replace('_', ' ')}: {tipo_estatistica} em {tamanho_streak} jogos {odds_text}"
        
        if bonus_complementar > 0:
            analise += f" [BONUS +{bonus_complementar:.0%} complementar]"
        
        analise += f" Previsao: {previsao.replace('VERDADEIRO', 'MANTERA STREAK').replace('FALSO', 'INTERROMPERA STREAK')}"
        
        if not has_odds or pd.isna(row['Odds']):
            analise += " [SEM ODDS - ANALISE QUALITATIVA]"
        
        return {
            'probabilidade': prob,
            'previsao': previsao,
            'padrao': padrao,
            'recomendacao': recomendacao,
            'analise': analise,
            'bonus': bonus_complementar
        }
    
    def _gerar_analise_texto(self, prob, previsao, tipo_estatistica, streak, odds, padrao, bonus, is_previsao=False):
        """Gerar análise textual"""
        
        base = f"{padrao.replace('_', ' ')}: "
        
        if is_previsao:
            base = f"PREVISAO - {base}"
        
        if tipo_estatistica == 'VITORIA':
            base += f"Streak de {streak} vitorias consecutivas "
        elif tipo_estatistica == 'DERROTA':
            base += f"Streak de {streak} derrotas consecutivas "
        elif tipo_estatistica == 'OVER_2.5':
            base += f"Over 2.5 goals em {streak} jogos "
        else:
            base += f"Estatistica {tipo_estatistica} por {streak} jogos "
        
        base += f"(Odds: {odds}). "
        
        if bonus > 0:
            base += f"[BONUS +{bonus:.0%} por estatisticas complementares] "
        
        if previsao == 1:
            base += "Previsao: MANTERA O STREAK."
        else:
            base += "Previsao: STREAK SERA INTERROMPIDO."
            
        if is_previsao:
            base += " [JOGO FUTURO]"
            
        return base
    
    def adicionar_analise_csv(self, output_path):
        """Adicionar colunas com análise ao CSV original - OTIMIZADO PARA FUTUROS"""
        probabilidades = []
        previsoes = []
        padroes = []
        recomendacoes = []
        analises_texto = []
        bonuses = []
        tipos_analise = []
        
        for idx, row in self.df.iterrows():
            # VERIFICAR SE É JOGO FUTURO (sem situação)
            has_situacao = 'Situacao' in self.df.columns and not pd.isna(row['Situacao']) and row['Situacao'] != ''
            is_futuro = not has_situacao
            
            if is_futuro:
                # VERIFICAR SE TEM ODDS
                has_odds = not pd.isna(row['Odds']) and row['Odds'] != ''
                
                if has_odds and self.model is not None:
                    # JOGO FUTURO COM ODDS - usar modelo completo
                    try:
                        resultado = self.analisar_jogo(
                            row['League'], row['Stat'], row['Next Match'], 
                            row['Odds'], row['Date']
                        )
                        probabilidades.append(resultado['Probabilidade_Sucesso'])
                        previsoes.append(resultado['Previsao'])
                        padroes.append(resultado['Padrao'])
                        recomendacoes.append(resultado['Recomendacao'])
                        analises_texto.append("PREVISAO - " + resultado['Analise_Texto'])
                        bonuses.append(resultado['Bonus_Complementar'])
                        tipos_analise.append('PREVISAO_FUTURO')
                        
                    except Exception as e:
                        # Fallback para análise básica
                        analise_basica = self._analise_basica_futuro(row, has_odds=True)
                        probabilidades.append(analise_basica['probabilidade'])
                        previsoes.append(analise_basica['previsao'])
                        padroes.append(analise_basica['padrao'])
                        recomendacoes.append(analise_basica['recomendacao'])
                        analises_texto.append("PREVISAO - " + analise_basica['analise'])
                        bonuses.append(analise_basica['bonus'])
                        tipos_analise.append('PREVISAO_BASICA')
                
                else:
                    # JOGO FUTURO SEM ODDS - análise qualitativa
                    analise_basica = self._analise_basica_futuro(row, has_odds=False)
                    probabilidades.append(analise_basica['probabilidade'])
                    previsoes.append(analise_basica['previsao'])
                    padroes.append(analise_basica['padrao'])
                    recomendacoes.append(analise_basica['recomendacao'])
                    analises_texto.append("PREVISAO - " + analise_basica['analise'])
                    bonuses.append(analise_basica['bonus'])
                    tipos_analise.append('PREVISAO_QUALITATIVA')
            
            else:
                # JOGO COM HISTÓRICO (já aconteceu)
                try:
                    resultado = self.analisar_jogo(
                        row['League'], row['Stat'], row['Next Match'], 
                        row['Odds'], row['Date']
                    )
                    probabilidades.append(resultado['Probabilidade_Sucesso'])
                    previsoes.append(resultado['Previsao'])
                    padroes.append(resultado['Padrao'])
                    recomendacoes.append(resultado['Recomendacao'])
                    analises_texto.append("HISTORICO - " + resultado['Analise_Texto'])
                    bonuses.append(resultado['Bonus_Complementar'])
                    tipos_analise.append('ANALISE_HISTORICA')
                except Exception as e:
                    probabilidades.append(None)
                    previsoes.append("ERRO_ANALISE")
                    padroes.append("INDETERMINADO")
                    recomendacoes.append("N/A")
                    analises_texto.append(f"Erro na analise: {str(e)}")
                    bonuses.append(0)
                    tipos_analise.append('ERRO')
        
        # Adicionar colunas
        self.df['Probabilidade_Sucesso'] = probabilidades
        self.df['Previsao_Modelo'] = previsoes
        self.df['Padrao_Identificado'] = padroes
        self.df['Recomendacao'] = recomendacoes
        self.df['Analise_Detalhada'] = analises_texto
        self.df['Bonus_Complementar'] = bonuses
        self.df['Tipo_Analise'] = tipos_analise
        
        # Salvar
        try:
            self.df.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')
        except:
            self.df.to_csv(output_path, index=False, sep=';', encoding='latin-1')
        
        print(f"CSV com analises salvo em: {output_path}")
        
        # Estatísticas finais
        total_jogos = len(self.df)
        jogos_futuros = sum(1 for t in tipos_analise if 'PREVISAO' in t)
        jogos_com_bonus = sum(1 for b in bonuses if b > 0)
        
        print(f"Total de jogos analisados: {total_jogos}")
        print(f"Jogos futuros com previsao: {jogos_futuros}")
        print(f"Jogos com bonus complementar: {jogos_com_bonus}")
        print(f"Jogos com historico: {total_jogos - jogos_futuros}")

# EXECUÇÃO PRINCIPAL
if __name__ == "__main__":
    # Inicializar analisador
    analisador = AnalisadorApostas('adam choi_dados_20251006_234547.csv')
    
    # Limpar dados
    analisador.limpar_dados()
    print("Dados limpos e preparados!")
    
    # Analisar jogos complementares
    analisador.analisar_jogos_complementares()
    
    # Treinar modelo (se houver dados históricos)
    analisador.treinar_modelo()
    
    # Analisar Kansas City especificamente
    try:
        kansas_analysis = analisador.analisar_jogo(
            league="US NWSL",
            stat="Kansas City Current W have Won their last 7 away league matches",
            next_match="Away vs Angel City W",
            odds=1.57,
            date="Monday, 06 October 23:30"
        )
        
        print("\n" + "="*60)
        print("ANALISE KANSAS CITY CURRENT W")
        print("="*60)
        print(f"Probabilidade: {kansas_analysis['Probabilidade_Sucesso']:.2%}")
        print(f"Previsao: {kansas_analysis['Previsao']}")
        print(f"Padrao: {kansas_analysis['Padrao']}")
        print(f"Recomendacao: {kansas_analysis['Recomendacao']}")
        print(f"Bonus: +{kansas_analysis['Bonus_Complementar']:.1%}")
        print(f"Analise: {kansas_analysis['Analise_Texto']}")
    except Exception as e:
        print(f"Erro na analise do Kansas City: {e}")
    
    # Salvar CSV completo
    analisador.adicionar_analise_csv('analise_completa_final.csv')
    
    print("\n" + "="*60)
    print("PROCESSO CONCLUIDO!")
    print("="*60)