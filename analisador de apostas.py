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
        """Limpar e preparar os dados para análise - OTIMIZADO"""
        # Converter datas - CORRIGINDO O ANO
        def corrigir_data(date_str):
            try:
                # Adicionar ano 2025 explicitamente para evitar warning
                date_with_year = f"{date_str} 2025"
                dt = pd.to_datetime(date_with_year, format='%A, %d %B %H:%M %Y', errors='coerce')
                return dt
            except:
                return pd.NaT
        
        self.df['Date'] = self.df['Date'].apply(corrigir_data)
        
        # Converter odds para numérico
        self.df['Odds'] = pd.to_numeric(self.df['Odds'], errors='coerce')
        
        # Criar target binário
        situacao_map = {
            'VERDADEIRO': 1, 'Verdadeiro': 1, 'VERDADEIR': 1,
            'FALSO': 0, 'Falso': 0, 'FALS': 0
        }
        
        if 'Situacao' in self.df.columns:
            self.df['Target'] = self.df['Situacao'].map(situacao_map).fillna(-1)
        else:
            self.df['Target'] = -1
        
        # Extrair características das estatísticas - MELHORADO
        self.df['Tipo_Estatistica'] = self.df['Stat'].apply(self._classificar_estatistica)
        self.df['Tamanho_Streak'] = self.df['Stat'].apply(self._extrair_streak)
        self.df['Local_Jogo'] = self.df['Next Match'].apply(self._extrair_local)
        self.df['Time'] = self.df['Stat'].apply(self._extrair_time)
        self.df['Liga_Categoria'] = self.df['League'].apply(self._classificar_liga)
        
        # NOVO: Calcular confiabilidade por liga
        self._calcular_confiabilidade_ligas()
        
        # Identificar jogos complementares
        self._identificar_jogos_complementares()
        
        # Dados para treino
        self.df_limpo = self.df[self.df['Target'] != -1].copy()
        
    def _classificar_liga(self, league):
        """Classificar liga por confiabilidade baseado na análise"""
        league_str = str(league).lower()
        
        # Baseado na análise dos registros
        if any(x in league_str for x in ['nwsl', 'women', 'feminino', 'feminina']):
            return 'ALTA_CONFIABILIDADE'
        elif any(x in league_str for x in ['norway', 'noruega', 'denmark', 'dinamarca', 'sweden', 'suecia']):
            return 'ALTA_CONFIABILIDADE'
        elif any(x in league_str for x in ['brasil', 'brazil', 'serie a', 'serie b']):
            return 'MEDIA_CONFIABILIDADE'
        elif any(x in league_str for x in ['mexico', 'argentina', 'chile']):
            return 'MEDIA_CONFIABILIDADE'
        else:
            return 'BAIXA_CONFIABILIDADE'
    
    def _calcular_confiabilidade_ligas(self):
        """Calcular confiabilidade real por liga baseada nos dados históricos"""
        if 'Target' in self.df.columns:
            ligas_confiabilidade = self.df[self.df['Target'] != -1].groupby('League').agg({
                'Target': 'mean',
                'Tipo_Estatistica': 'count'
            }).round(3)
            
            self.ligas_confiabilidade = ligas_confiabilidade
            print("\nCONFIABILIDADE POR LIGA (base real):")
            print(ligas_confiabilidade.sort_values('Target', ascending=False).head(10))
    
    def _extrair_time(self, stat):
        """Extrair nome do time da estatística"""
        try:
            return stat.split(' have ')[0].strip()
        except:
            return 'TIME_DESCONHECIDO'
    
    def _identificar_jogos_complementares(self):
        """Identificar jogos com estatísticas complementares - MELHORADO"""
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
        
        # Filtrar jogos com múltiplas estatísticas
        self.jogos_complementares = {}
        
        for chave, jogo in jogos_unicos.items():
            if len(jogo['stats']) >= 2:
                times = [s['time'] for s in jogo['stats']]
                if len(set(times)) >= 2:
                    self.jogos_complementares[chave] = jogo
        
        print(f"Jogos com estatisticas complementares: {len(self.jogos_complementares)}")
    
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
    
    def _classificar_padrao(self, probabilidade, bonus_complementar=0):
        """Classificar padrão baseado na probabilidade - AJUSTADO"""
        prob_ajustada = probabilidade + bonus_complementar
        
        # Novos thresholds baseados na análise
        if prob_ajustada > 0.78:
            return 'PADRAO_FORTISSIMO'
        elif prob_ajustada > 0.68:
            return 'PADRAO_FORTE'
        elif prob_ajustada > 0.58:
            return 'PADRAO_SOLIDO'
        else:
            return 'PADRAO_REGULAR'
    
    def _classificar_recomendacao(self, probabilidade, bonus_complementar=0):
        """Classificar recomendação - AJUSTADO"""
        prob_ajustada = probabilidade + bonus_complementar
        
        if prob_ajustada > 0.75:
            return 'EXCELENTE'
        elif prob_ajustada > 0.62:
            return 'BOA'
        else:
            return 'REGULAR'
    
    def _calcular_bonus_complementar(self, league, next_match, date):
        """Calcular bônus para jogos complementares - MELHORADO"""
        chave_jogo = f"{date}_{league}"
        
        if chave_jogo in self.jogos_complementares:
            jogo = self.jogos_complementares[chave_jogo]
            tipos_stats = [s['tipo'] for s in jogo['stats']]
            
            # Bônus aumentados baseados na análise
            if 'VITORIA' in tipos_stats and 'DERROTA' in tipos_stats:
                return 0.18
            elif 'OVER_2.5' in tipos_stats and 'BTTS' in tipos_stats:
                return 0.15
            elif len(set(tipos_stats)) >= 2:
                return 0.10
            
        return 0.0
    
    def _calcular_bonus_confiabilidade_liga(self, league):
        """Calcular bônus baseado na confiabilidade da liga"""
        liga_str = str(league).lower()
        
        if any(x in liga_str for x in ['nwsl', 'women', 'norway', 'denmark']):
            return 0.08
        elif any(x in liga_str for x in ['brasil', 'sweden', 'iceland']):
            return 0.05
        else:
            return 0.0
    
    def analisar_jogos_complementares(self):
        """Analisar jogos complementares - MELHORADO"""
        if not self.jogos_complementares:
            print("Nenhum jogo complementar identificado")
            return
        
        resultados = []
        
        for chave, jogo in self.jogos_complementares.items():
            situacoes = [s['situacao'] for s in jogo['stats'] if s['situacao'] in ['VERDADEIRO', 'Verdadeiro', 'FALSO', 'Falso']]
            
            if situacoes:
                acertos = sum(1 for s in situacoes if s in ['VERDADEIRO', 'Verdadeiro'])
                taxa_acerto = acertos / len(situacoes)
            else:
                taxa_acerto = None
            
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
        
        df_resultados = pd.DataFrame(resultados)
        
        if not df_resultados.empty:
            print("\n" + "="*80)
            print("ANALISE DE JOGOS COM ESTATISTICAS COMPLEMENTARES")
            print("="*80)
            
            jogos_com_resultado = df_resultados[df_resultados['Taxa_Acerto'].notna()]
            
            if not jogos_com_resultado.empty:
                taxa_media = jogos_com_resultado['Taxa_Acerto'].mean()
                print(f"Taxa media de acerto em jogos complementares: {taxa_media:.1%}")
                print(f"Total de jogos analisados: {len(jogos_com_resultado)}")
            
            combinacoes = df_resultados['Combinacao_Stats'].value_counts()
            print(f"\nCOMBINACOES MAIS FREQUENTES:")
            for combinacao, count in combinacoes.head(5).items():
                print(f"  {combinacao}: {count} jogos")
            
            df_resultados.to_csv('analise_jogos_complementares.csv', index=False, encoding='utf-8-sig')
            print(f"\nAnalise detalhada salva em: analise_jogos_complementares.csv")
            
            return df_resultados
    
    def treinar_modelo(self):
        """Treinar modelo de machine learning - OTIMIZADO"""
        if len(self.df_limpo) < 15:
            print("Dados insuficientes para treinar modelo. Usando analise qualitativa.")
            self.model = None
            return
        
        # NOVAS FEATURES
        features = ['Odds', 'Tamanho_Streak', 'Tipo_Estatistica', 'Local_Jogo', 'Liga_Categoria']
        
        # Codificar variáveis
        X = self.df_limpo[features].copy()
        X['Tipo_Estatistica'] = self.encoder.fit_transform(X['Tipo_Estatistica'])
        X['Local_Jogo'] = LabelEncoder().fit_transform(X['Local_Jogo'])
        X['Liga_Categoria'] = LabelEncoder().fit_transform(X['Liga_Categoria'])
        
        y = self.df_limpo['Target']
        
        # Treinar com mais árvores
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10)
        self.model.fit(X_train, y_train)
        
        accuracy = self.model.score(X_test, y_test)
        print(f"Acuracia do modelo OTIMIZADO: {accuracy:.2%}")
        
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nImportancia das variaveis OTIMIZADO:")
        print(feature_importance)
    
    def analisar_jogo(self, league, stat, next_match, odds, date):
        """Analisar jogo específico - MELHORADO"""
        if self.model is None:
            return self._analise_basica_futuro({
                'League': league, 'Stat': stat, 'Next Match': next_match,
                'Odds': odds, 'Date': date
            }, has_odds=not pd.isna(odds))
        
        tipo_estatistica = self._classificar_estatistica(stat)
        tamanho_streak = self._extrair_streak(stat)
        local_jogo = self._extrair_local(next_match)
        liga_categoria = self._classificar_liga(league)
        
        # CALCULAR TODOS OS BÔNUS
        bonus_complementar = self._calcular_bonus_complementar(league, next_match, date)
        bonus_confiabilidade = self._calcular_bonus_confiabilidade_liga(league)
        bonus_total = bonus_complementar + bonus_confiabilidade
        
        # Criar feature vector
        features = pd.DataFrame([{
            'Odds': odds,
            'Tamanho_Streak': tamanho_streak,
            'Tipo_Estatistica': tipo_estatistica,
            'Local_Jogo': local_jogo,
            'Liga_Categoria': liga_categoria
        }])
        
        # Codificar
        features['Tipo_Estatistica'] = self.encoder.transform(features['Tipo_Estatistica'])
        features['Local_Jogo'] = LabelEncoder().fit_transform(features['Local_Jogo'])
        features['Liga_Categoria'] = LabelEncoder().fit_transform(features['Liga_Categoria'])
        
        # Prever
        probabilidade = self.model.predict_proba(features)[0][1]
        previsao = self.model.predict(features)[0]
        
        # Aplicar todos os bônus
        probabilidade_ajustada = min(probabilidade + bonus_total, 0.95)
        
        padrao = self._classificar_padrao(probabilidade, bonus_total)
        recomendacao = self._classificar_recomendacao(probabilidade, bonus_total)
        
        analise = self._gerar_analise_texto(probabilidade_ajustada, previsao, tipo_estatistica, 
                                          tamanho_streak, odds, padrao, bonus_total, 
                                          is_previsao=True, liga=league)
        
        return {
            'Probabilidade_Sucesso': probabilidade_ajustada,
            'Previsao': 'VERDADEIRO' if previsao == 1 else 'FALSO',
            'Padrao': padrao,
            'Recomendacao': recomendacao,
            'Analise_Texto': analise,
            'Bonus_Complementar': bonus_complementar,
            'Bonus_Confiabilidade': bonus_confiabilidade,
            'Bonus_Total': bonus_total
        }
    
    def _analise_basica_futuro(self, row, has_odds=True):
        """Análise básica para futuros - MELHORADA"""
        tipo_estatistica = self._classificar_estatistica(row['Stat'])
        tamanho_streak = self._extrair_streak(row['Stat'])
        local_jogo = self._extrair_local(row['Next Match'])
        
        bonus_complementar = self._calcular_bonus_complementar(
            row['League'], row['Next Match'], row['Date']
        )
        bonus_confiabilidade = self._calcular_bonus_confiabilidade_liga(row['League'])
        bonus_total = bonus_complementar + bonus_confiabilidade
        
        # PROBABILIDADES BASE AJUSTADAS
        prob_base = {
            'VITORIA': 0.72 if tamanho_streak >= 6 else 0.62,
            'DERROTA': 0.70 if tamanho_streak >= 6 else 0.60,
            'OVER_2.5': 0.65,
            'BTTS': 0.63,
            'EMPATE': 0.48,
            'OUTRO': 0.55
        }
        
        prob = prob_base.get(tipo_estatistica, 0.55) + bonus_total
        prob = min(prob, 0.90)
        
        # CLASSIFICAÇÃO MELHORADA
        if prob > 0.75:
            padrao = 'PADRAO_FORTE'
            recomendacao = 'EXCELENTE'
        elif prob > 0.65:
            padrao = 'PADRAO_SOLIDO'
            recomendacao = 'BOA'
        else:
            padrao = 'PADRAO_REGULAR'
            recomendacao = 'REGULAR'
        
        previsao = 'VERDADEIRO'
        
        odds_text = f"(Odds: {row['Odds']})" if has_odds and not pd.isna(row['Odds']) else "(Odds: N/D)"
        
        analise = f"{padrao}: {tipo_estatistica} em {tamanho_streak} jogos {odds_text}"
        
        if bonus_total > 0:
            analise += f" [BONUS +{bonus_total:.0%} complementar+confiabilidade]"
        
        analise += f" Previsao: {previsao.replace('VERDADEIRO', 'MANTERA STREAK')}"
        
        if not has_odds or pd.isna(row['Odds']):
            analise += " [SEM ODDS - ANALISE QUALITATIVA]"
        
        return {
            'probabilidade': prob,
            'previsao': previsao,
            'padrao': padrao,
            'recomendacao': recomendacao,
            'analise': analise,
            'bonus': bonus_total
        }
    
    def _gerar_analise_texto(self, prob, previsao, tipo_estatistica, streak, odds, padrao, bonus, is_previsao=False, liga=None):
        """Gerar análise textual - MELHORADA"""
        
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
            base += f"[BONUS +{bonus:.0%} complementar+confiabilidade] "
        
        if liga and any(x in str(liga).lower() for x in ['nwsl', 'women', 'norway']):
            base += "[LIGA ALTA CONFIABILIDADE] "
        
        if previsao == 1:
            base += "Previsao: MANTERA O STREAK."
        else:
            base += "Previsao: STREAK SERA INTERROMPIDO."
            
        if is_previsao:
            base += " [JOGO FUTURO]"
            
        return base

    def adicionar_analise_csv(self, output_path):
        """Adicionar colunas com análise ao CSV original - COMPLETO"""
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
                        bonuses.append(resultado['Bonus_Total'])
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
                    bonuses.append(resultado['Bonus_Total'])
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
        self.df['Bonus_Total'] = bonuses
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
        print(f"Jogos com bonus: {jogos_com_bonus}")
        print(f"Jogos com historico: {total_jogos - jogos_futuros}")

# EXECUÇÃO PRINCIPAL
if __name__ == "__main__":
    # ⬇️⬇️⬇️ APONTANDO PARA O ARQUIVO ATUALIZADO ⬇️⬇️⬇️
    analisador = AnalisadorApostas('adam choi_dados_20251005_000421.csv')
    
    # Limpar dados
    analisador.limpar_dados()
    print("Dados limpos e preparados!")
    
    # Analisar jogos complementares
    analisador.analisar_jogos_complementares()
    
    # Treinar modelo
    analisador.treinar_modelo()
    
    # Salvar análise completa
    analisador.adicionar_analise_csv('analise_completa_atualizada.csv')
    
    print("\n" + "="*60)
    print("PROCESSO CONCLUIDO!")
    print("="*60)