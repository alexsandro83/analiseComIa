import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
from datetime import datetime, timedelta
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class AnalisadorApostasEvolutivo:
    def __init__(self, base_treino_path=None, base_futuros_path=None, modelo_path='modelo_apostas.joblib'):
        self._instalar_dependencias()
        self.base_treino_path = base_treino_path
        self.base_futuros_path = base_futuros_path
        self.modelo_path = modelo_path
        self.model = None
        self.encoder = LabelEncoder()
        self.jogos_complementares = {}
        self.features_para_treino = ['Odds', 'Tamanho_Streak', 'Tipo_Estatistica', 'Local_Jogo', 'Liga_Categoria']
        self.debug_mode = True  # ✅ Adicionar modo debug
    def verificar_arquivos_config(self, config):
        """Verificar se arquivos de configuração existem"""
        arquivos_ok = True
        
        for nome, caminho in config.items():
            if 'base' in nome and not os.path.exists(caminho):
                print(f"❌ Arquivo não encontrado: {caminho}")
                arquivos_ok = False
            else:
                print(f"✅ {nome}: {caminho}")
        
        return arquivos_ok
    def _instalar_dependencias(self):
        """Instalar psutil automaticamente se necessário"""
        try:
            import psutil
            return True
        except ImportError:
            self._log_detalhado("📦 Instalando psutil...", "INFO")
            try:
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
                self._log_detalhado("✅ psutil instalado com sucesso", "SUCESSO")
                return True
            except Exception as e:
                self._log_detalhado(f"❌ Falha ao instalar psutil: {e}", "ERRO")
                return False

    def _log_detalhado(self, mensagem, nivel="INFO"):
        """Sistema de logging organizado"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        icones = {"INFO": "ℹ️", "ERRO": "❌", "SUCESSO": "✅", "ALERTA": "⚠️"}
        icone = icones.get(nivel, "🔍")
        print(f"{mensagem}")
    def _calcular_efetividade(self, situacao, previsao, date_str):
        """Calcular efetividade considerando data do jogo + 2 horas de margem"""
        
        
        situacao_clean = str(situacao).strip()
        previsao_clean = str(previsao).strip()
        date_clean = str(date_str).strip()
        
        # Verificar se é data inválida
        if date_clean in ['', 'nan', 'None', 'DATA_INDISPONIVEL', 'NaT']:
            return 'DATA_INVALIDA'
        
        # Verificar se o jogo já aconteceu (data + 2 horas)
        try:
            # Converter data do formato "25/10/2025 20:30"
            data_jogo = datetime.strptime(date_clean, '%d/%m/%Y %H:%M')
            agora = datetime.now()
            
            # Adicionar 2 horas de margem (jogo já terminou)
            limite_verificacao = data_jogo + timedelta(hours=2)
            #breakpoint()
            # Se ainda não passou 2 horas do início do jogo
            if agora < limite_verificacao:
                return 'JOGO_FUTURO'
            elif ((limite_verificacao - data_jogo) < timedelta(hours=2) and agora > data_jogo):
                return 'JOGO_EM_ANDAMENTO'
            else:
                # Jogo já terminou (passou mais de 2 horas do início)
                if situacao_clean == '':
                    return 'AGUARDANDO_RESULTADO'
                elif previsao_clean == '':
                    return 'SEM_PREVISAO'
                elif situacao_clean.upper() == previsao_clean.upper():
                    return 'ACERTO'
                else:
                    return 'ERRO'
                    
        except Exception as e:
            self._log_detalhado(f"Erro ao processar data '{date_clean}': {e}", "ALERTA")
            return 'ERRO_DATA'
    
    def _adicionar_coluna_efetividade(self, df):
        """Adicionar coluna de efetividade ao DataFrame - VERSÃO CORRIGIDA"""
        if 'Situação' in df.columns and 'Previsao' in df.columns and 'Date' in df.columns:
            df['Efetividade'] = df.apply(
                lambda row: self._calcular_efetividade(
                    row['Situação'], 
                    row['Previsao'],
                    row['Date']  # ⬅️ AGORA COM O date_str!
                ), 
                axis=1
            )
            
            # Estatísticas detalhadas
            stats = df['Efetividade'].value_counts()
            self._log_detalhado("📊 ESTATÍSTICAS DE EFETIVIDADE:")
            for status, count in stats.items():
                percentual = (count / len(df)) * 100
                self._log_detalhado(f"   {status}: {count} jogos ({percentual:.1f}%)")
                
        else:
            df['Efetividade'] = 'COLUNAS_INCOMPLETAS'
            self._log_detalhado("Colunas Situação, Previsao ou Date não encontradas", "ALERTA")
        
        return df
    def _calcular_estatisticas_efetividade(self, df):
        """Calcular estatísticas de efetividade"""
        if 'Efetividade' not in df.columns:
            return "Estatísticas não disponíveis"
        
        stats = df['Efetividade'].value_counts()
        total = len(df)
        acertos = stats.get('ACERTO', 0)
        erros = stats.get('ERRO', 0)
        
        if total > 0:
            taxa_acerto = (acertos / total) * 100
            return f"Acertos: {acertos}/{total} ({taxa_acerto:.1f}%)"
        else:
            return "Nenhum dado para análise"    
    def _log_detalhado(self, mensagem, nivel="INFO"):
        """Sistema de logging organizado"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        icones = {"INFO": "ℹ️", "ERRO": "❌", "SUCESSO": "✅", "ALERTA": "⚠️"}
        icone = icones.get(nivel, "🔍")
        # print(f"{timestamp} {icone} {mensagem}")
        print(f"{mensagem}")
    def _fazer_backup_modelo(self, nova_acuracia=None):
        """Criar backup do modelo apenas se a acurácia melhorar"""
        if not os.path.exists(self.modelo_path):
            self._log_detalhado("Nenhum modelo existente para backup", "ALERTA")
            return False
        
        try:
            # Carregar modelo atual para comparar acurácia
            modelo_atual = joblib.load(self.modelo_path)
            acuracia_atual = modelo_atual.get('acuracia', 0)
            
            # Se nova acurácia não foi fornecida, verificar se temos no objeto
            if nova_acuracia is None:
                if hasattr(self, 'acuracia_modelo'):
                    nova_acuracia = self.acuracia_modelo
                else:
                    # Se não tem acurácia nova, faz backup padrão (para backup manual)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = f"{self.modelo_path}.backup_{timestamp}"
                    import shutil
                    shutil.copy2(self.modelo_path, backup_path)
                    self._log_detalhado(f"✅ Backup manual criado: {backup_path}", "SUCESSO")
                    return True
            
            # Só faz backup se a acurácia melhorou (para treinamento automático)
            if nova_acuracia > acuracia_atual:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{self.modelo_path}.backup_{timestamp}_acc{nova_acuracia:.3f}"
                import shutil
                shutil.copy2(self.modelo_path, backup_path)
                self._log_detalhado(f"✅ Backup criado (melhoria: {acuracia_atual:.2%} → {nova_acuracia:.2%}): {backup_path}", "SUCESSO")
                return True
            else:
                self._log_detalhado(f"⚠️  Backup não necessário (acurácia: {acuracia_atual:.2%} → {nova_acuracia:.2%})", "ALERTA")
                return False
                
        except Exception as e:
            self._log_detalhado(f"❌ Erro ao verificar acurácia para backup: {e}", "ERRO")
            return False
    def _validar_dados_treino(self):
        """Validar integridade dos dados de treino"""
        colunas_essenciais = ['Situação', 'Odds', 'Stat']
        for col in colunas_essenciais:
            if col not in self.df_treino.columns:
                raise ValueError(f"Coluna essencial '{col}' não encontrada")
        
        # Verificar valores nulos
        nulos = self.df_treino[colunas_essenciais].isnull().sum()
        if nulos.any():
            self._log_detalhado(f"Valores nulos encontrados: {nulos}", "ALERTA")
        
        # ✅ NOVA VERIFICAÇÃO: Garantir que há dados suficientes
        if len(self.df_treino) < 10:
            raise ValueError(f"Dados insuficientes: apenas {len(self.df_treino)} registros")
    
    def _extrair_time_avancado(self, stat):
            """Extrair nome do time com múltiplos padrões - VERSÃO MELHORADA"""
            stat_str = str(stat).strip()
            
            # DEBUG: Log para ver o que está sendo processado
            if 'TIME_DESCONHECIDO' in stat_str:
                self._log_detalhado(f"🔍 DEBUG Stat problemático: '{stat_str}'", "ALERTA")
            
            # Padrões principais em ordem de prioridade
            padroes = [
                # Padrão: "Team have/has/had ..."
                (r'^([A-Za-z\s\-\'\.]+) (?:have|has|had) (?:won|lost|drawn|scored|conceded|played)'),
                # Padrão: "Team won/lost/drew ..."
                (r'^([A-Za-z\s\-\'\.]+) (?:won|lost|drew)'),
                # Padrão: "Team in ..."
                (r'^([A-Za-z\s\-\'\.]+) in'),
                # Padrão: "Team ... matches"
                (r'^([A-Za-z\s\-\'\.]+) (?:last|previous|recent)'),
                # Padrão: "Team's ..."
                (r'^([A-Za-z\s\-\'\.]+)\'s'),
                # Padrão: "Team ... games"
                (r'^([A-Za-z\s\-\'\.]+) (?:games|matches|fixtures)'),
            ]
            
            for padrao in padroes:
                match = re.search(padrao, stat_str, re.IGNORECASE)
                if match:
                    time = match.group(1).strip()
                    # Limpeza do nome
                    time = re.sub(r'\s+', ' ', time)
                    time=re.split(r' have', time, flags=re.IGNORECASE)[0]
                    # Remove espaços múltiplos
                    time = time.title()  # Primeira letra maiúscula em cada palavra
                    
                    # Verificar se é um nome válido (não muito curto e não contém palavras comuns)
                    palavras_invalidas = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
                    palavras_time = time.split()
                    if (len(time) >= 3 and 
                        not any(palavra.lower() in palavras_invalidas for palavra in palavras_time) and
                        len(palavras_time) <= 4):  # Máximo de 4 palavras
                        self._log_detalhado(f"✅ Time extraído: '{time}' do padrão: {padrao}", "SUCESSO")
                        return time
            
            # Se nenhum padrão funcionou, tentar fallback
            time_fallback = self._extrair_time_fallback(stat_str)
            if time_fallback != 'TIME_DESCONHECIDO':
                return time_fallback
            
            self._log_detalhado(f"❌ Não foi possível extrair time de: '{stat_str}'", "ALERTA")
            return 'TIME_DESCONHECIDO'
    def _extrair_time_fallback(self, stat_str):
        """Fallback para extração de times - casos especiais"""
        # Caso 1: Dividir por "have", "has", "had"
        for separador in [' have ', ' has ', ' had ', ' won ', ' lost ', ' drew ']:
            if separador in stat_str:
                partes = stat_str.split(separador)
                if len(partes) > 1:
                    time = partes[0].strip()
                    if len(time) >= 2:
                        return time.title()
        
        # Caso 2: Procurar por nomes de times conhecidos no início
        times_comuns = [
            'Crvena Zvezda', 'Den Haag', 'Rosengard', 'Barcelona', 'Real Madrid', 
            'Manchester', 'Liverpool', 'Chelsea', 'Arsenal', 'Bayern', 'PSG'
        ]
        
        for time_comum in times_comuns:
            if stat_str.startswith(time_comum):
                return time_comum
        
        # Caso 3: Primeiras 2-3 palavras como fallback
        palavras = stat_str.split()
        if len(palavras) >= 2:
            # Tentar com 2 palavras
            time_candidato = ' '.join(palavras[:2])
            if len(time_candidato) >= 4:
                return time_candidato.title()
        
        return 'TIME_DESCONHECIDO'
    def _preparar_dados_treino(self):
        """Preparar dados para treino - VERSÃO CORRIGIDA"""
        self._log_detalhado("Preparando dados para treino...")

        # DEBUG: Mostrar informações dos dados
        self._log_detalhado(f"Colunas disponíveis: {list(self.df_treino.columns)}")
        self._log_detalhado(f"Primeiras linhas:")
        print(self.df_treino.head(2))
        
        # ✅ CORREÇÃO: REMOVER coluna Resultado para evitar confusão
        if 'Resultado' in self.df_treino.columns:
            self.df_treino = self.df_treino.drop('Resultado', axis=1)
            self._log_detalhado("✅ Coluna 'Resultado' removida - não usada no modelo")
        
        def corrigir_data(date_str):
            try:
                # Sua base já tem data no formato correto: "27/11/2025 20:30"
                return pd.to_datetime(date_str, format='%d/%m/%Y %H:%M', errors='coerce')
            except:
                try:
                    # Tentativa alternativa
                    return pd.to_datetime(date_str, errors='coerce')
                except:
                    return pd.NaT
        
        # Processar Date
        if 'Date' in self.df_treino.columns:
            self.df_treino['Date'] = self.df_treino['Date'].apply(corrigir_data)
            self._log_detalhado(f"Datas processadas: {self.df_treino['Date'].notna().sum()} válidas")
        else:
            self._log_detalhado("Coluna 'Date' não encontrada", "ALERTA")
            self.df_treino['Date'] = pd.NaT
        
        # Processar Odds
        if 'Odds' in self.df_treino.columns:
            self.df_treino['Odds'] = pd.to_numeric(self.df_treino['Odds'], errors='coerce')
            self._log_detalhado(f"Odds processadas: {self.df_treino['Odds'].notna().sum()} válidas")
        else:
            self._log_detalhado("Coluna 'Odds' não encontrada - ESSENCIAL", "ERRO")
            return
        
        # CORREÇÃO: Usar coluna 'Situação' que existe na sua base
        if 'Situação' in self.df_treino.columns:
            situacao_map = {'VERDADEIRO': 1, 'Verdadeiro': 1, 'FALSO': 0, 'Falso': 0, 'GREEN': 1, 'RED': 0, 'WIN': 1, 'LOSS': 0}
            self.df_treino['Target'] = self.df_treino['Situação'].map(situacao_map).fillna(-1)
            self._log_detalhado(f"Target criado: {len(self.df_treino[self.df_treino['Target'] != -1])} amostras válidas")
        else:
            self._log_detalhado("Coluna 'Situação' não encontrada - ESSENCIAL", "ERRO")
            return
        
        # Features - verificar colunas essenciais
        if 'Stat' in self.df_treino.columns:
            self.df_treino['Tipo_Estatistica'] = self.df_treino['Stat'].apply(self._classificar_estatistica)
            self.df_treino['Tamanho_Streak'] = self.df_treino['Stat'].apply(self._extrair_streak)
            self._log_detalhado(f"Stats processados: {len(self.df_treino)} registros")
        else:
            self._log_detalhado("Coluna 'Stat' não encontrada - ESSENCIAL", "ERRO")
            return
        
        if 'Next Match' in self.df_treino.columns:
            self.df_treino['Local_Jogo'] = self.df_treino['Next Match'].apply(self._extrair_local)
            self._log_detalhado("Local do jogo processado")
        else:
            self._log_detalhado("Coluna 'Next Match' não encontrada", "ALERTA")
            self.df_treino['Local_Jogo'] = 'NEUTRO'
        
        if 'League' in self.df_treino.columns:
            self.df_treino['Liga_Categoria'] = self.df_treino['League'].apply(self._classificar_liga)
            self._log_detalhado("Liga categorizada")
        else:
            self._log_detalhado("Coluna 'League' não encontrada", "ALERTA")
            self.df_treino['Liga_Categoria'] = 'MEDIA_CONFIABILIDADE'

        # ✅ CORREÇÃO: Filtrar apenas dados com target válido ANTES de criar df_treino_limpo
        self.df_treino_limpo = self.df_treino[self.df_treino['Target'] != -1].copy()
        self._log_detalhado(f"Dados finais para treino: {len(self.df_treino_limpo)} registros válidos")
        
        # ✅ CORREÇÃO: AGORA criar as features de engenharia APÓS ter df_treino_limpo
        if len(self.df_treino_limpo) > 0:
            # 1. Interação entre Odds e Streak
            self.df_treino_limpo['Odds_Streak_Interaction'] = (
                self.df_treino_limpo['Odds'] * self.df_treino_limpo['Tamanho_Streak']
            )
            
            # 2. Probabilidade implícita das Odds
            def categorizar_streak(streak):
                if streak <= 3:
                    return 'Curto'
                elif streak <= 6:
                    return 'Medio'
                elif streak <= 10:
                    return 'Longo'
                else:
                    return 'Muito_Longo'
            
            self.df_treino_limpo['Streak_Categoria'] = self.df_treino_limpo['Tamanho_Streak'].apply(
                lambda x: categorizar_streak(x) if pd.notnull(x) else None
            )
            
            # 4. Dia da semana (padrões temporais)
            if 'Date' in self.df_treino_limpo.columns:
                self.df_treino_limpo['Dia_Semana'] = self.df_treino_limpo['Date'].dt.dayofweek
            
            # Atualizar lista de features
            self.features_para_treino.extend([
                'Odds_Streak_Interaction', 
                'Prob_Implicita',
                'Streak_Categoria',
                'Dia_Semana'
            ])
            
            # Mostrar distribuição
            verd_count = (self.df_treino_limpo['Target'] == 1).sum()
            fals_count = (self.df_treino_limpo['Target'] == 0).sum()
            self._log_detalhado(f"Distribuição: VERDADEIRO={verd_count}, FALSO={fals_count}")
        
        # ✅ ADICIONAR APÓS CRIAR A COLUNA TARGET:
        # Adicionar coluna de efetividade para dados de treino
        self.df_treino = self._adicionar_coluna_efetividade(self.df_treino)
        
        # ✅ NOVA VERIFICAÇÃO: Garantir que temos dados suficientes após todo o processamento
        if len(self.df_treino_limpo) < 10:
            self._log_detalhado(f"⚠️  AVISO: Apenas {len(self.df_treino_limpo)} registros válidos após processamento", "ALERTA")
        else:
            self._log_detalhado(f"✅ Dados preparados com sucesso: {len(self.df_treino_limpo)} registros válidos", "SUCESSO")
    def corrigir_caracteres_especiais_csv(self, caminho_arquivo, df):
        """Corrigir caracteres especiais - VERSÃO CORRIGIDA"""
        # Mapeamento completo de correção de caracteres especiais
        correcoes = {
            'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú', 'Ã±': 'ñ',
            'Ã£': 'ã', 'Ãµ': 'õ', 'Ã§': 'ç', 'Ã¼': 'ü', 'Ã¶': 'ö', 'Ã¤': 'ä',
            'Ã¸': 'ø', 'Ã¦': 'æ', 'Ã…': 'Å', 'Ã€': 'À', 'Ã‚': 'Â', 'Ãƒ': 'Ã',
            'Ã„': 'Ä', 'Ã‡': 'Ç', 'Ãˆ': 'È', 'Ã‰': 'É', 'ÃŠ': 'Ê', 'Ã‹': 'Ë',
            'ÃŒ': 'Ì', 'Ã': 'Í', 'ÃŽ': 'Î', 'Ã‘': 'Ñ', 'Ã’': 'Ò', 'Ã“': 'Ó',
            'Ã”': 'Ô', 'Ã•': 'Õ', 'Ã–': 'Ö', 'Ã—': '×', 'Ã˜': 'Ø', 'Ã™': 'Ù',
            'Ãš': 'Ú', 'Ã›': 'Û', 'Ãœ': 'Ü', 'Ã': 'Ý', 'Ãž': 'Þ', 'ÃŸ': 'ß',
            'Â': '', 'â': '', '€': '', '': '', '': '', '': '', '': '',
            '': '', '': '', '': '', '': '', '': '', '': '', '': '',
            '': '', '': '', '': '', '': '', '': '', '': '', '': '',
            '': '', '': '', '¡': '', '¢': '', '£': '', '¤': '', '¥': '',
            '¦': '', '§': '', '¨': '', '©': '', 'ª': '', '«': '', '¬': '',
            '­': '', '®': '', '¯': '', '°': '', '±': '', '²': '', '³': '',
            '´': '', 'µ': '', '¶': '', '·': '', '¸': '', '¹': '', 'º': '',
            '»': '', '¼': '', '½': '', '¾': '', '¿': '', 'À': 'A', 'Á': 'A',
            'Â': 'A', 'Ã': 'A', 'Ä': 'A', 'Å': 'A', 'Æ': 'AE', 'Ç': 'C',
            'È': 'E', 'É': 'E', 'Ê': 'E', 'Ë': 'E', 'Ì': 'I', 'Í': 'I',
            'Î': 'I', 'Ï': 'I', 'Ð': 'D', 'Ñ': 'N', 'Ò': 'O', 'Ó': 'O',
            'Ô': 'O', 'Õ': 'O', 'Ö': 'O', 'Ø': 'O', 'Ù': 'U', 'Ú': 'U',
            'Û': 'U', 'Ü': 'U', 'Ý': 'Y', 'Þ': 'TH', 'ß': 'ss', 'à': 'a',
            'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a', 'æ': 'ae',
            'ç': 'c', 'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e', 'ì': 'i',
            'í': 'i', 'î': 'i', 'ï': 'i', 'ð': 'd', 'ñ': 'n', 'ò': 'o',
            'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'ø': 'o', 'ù': 'u',
            'ú': 'u', 'û': 'u', 'ü': 'u', 'ý': 'y', 'þ': 'th', 'ÿ': 'y'
        }

        # ✅ 1. PRIMEIRO: Aplicar correções em todas as colunas de texto
        for coluna in df.columns:
            if df[coluna].dtype == 'object':
                for errado, correto in correcoes.items():
                    df[coluna] = df[coluna].str.replace(errado, correto, regex=False)

        # ✅ 2. SEGUNDO: Limpar espaços e remover duplicatas
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
        # Verificar se as colunas existem antes de remover duplicatas
        colunas_subset = ['League', 'Stat', 'Next Match', 'Date', 'Situação']
        colunas_existentes = [col for col in colunas_subset if col in df.columns]
        
        if len(colunas_existentes) >= 3:  # Mínimo de colunas para considerar únicas
            df = df.drop_duplicates(subset=colunas_existentes, keep='first')
        
        # ✅ 3. TERCEIRO: Ordenar os dados
        colunas_ordenacao = []
        if 'Date' in df.columns:
            colunas_ordenacao.append('Date')
        if 'Stat' in df.columns:
            colunas_ordenacao.append('Stat')
        
        if colunas_ordenacao:
            df = df.sort_values(colunas_ordenacao, ascending=[False, False])

        # ✅ 4. QUARTO: SALVAR APÓS todas as transformações
            df = df.sort_values('Odds', ascending=False).drop_duplicates(
            subset=['League', 'Stat', 'Next Match', 'Date'], 
            keep='first'
        ).sort_values('Date', ascending=False).reset_index(drop=True)
        text_columns = ['League', 'Stat', 'Next Match', 'Odds', 'Situação',
       'Tipo_Estatistica', 'Liga_Categoria', 'Probabilidade_Sucesso',
       'Efetividade', 'Previsao', 'Padrao', 'Recomendacao',
       'Analise_Detalhada']
        for col in text_columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

        
        df.to_csv(caminho_arquivo, index=False, sep=';', encoding='utf-8-sig')
        
        self._log_detalhado(f"✅ Arquivo corrigido e salvo: {caminho_arquivo}", "SUCESSO")
        self._log_detalhado(f"   - Registros processados: {len(df)}")
        self._log_detalhado(f"   - Colunas: {list(df.columns)}")
        
        return df
    def carregar_dados(self, csv_path):
        try:
            
            # PRIMEIRO: Verificar o formato real do arquivo
            self._log_detalhado(f"Tentando carregar: {csv_path}")
            
            # Lê as primeiras linhas para debug
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                primeira_linha = f.readline().strip()
                segunda_linha = f.readline().strip()
                self._log_detalhado(f"Primeira linha: {primeira_linha}")
                self._log_detalhado(f"Segunda linha: {segunda_linha}")
            
            # Tenta diferentes delimitadores
            delimitadores = [';', ',', '\t', '|']
            
            for delim in delimitadores:
                try:
                    self._log_detalhado(f"Tentando delimitador: '{delim}'")
                    df = pd.read_csv(csv_path, delimiter=delim, encoding='utf-8-sig')
                    # Limpar espaços em todas as colunas de texto antes de remover duplicatas
                    
                    self._log_detalhado(f"Sucesso com delimitador '{delim}': {len(df)} linhas, {len(df.columns)} colunas")
                    self._log_detalhado(f"Colunas: {list(df.columns)}")
                    
                    if len(df.columns) > 1:
                        return df
                except Exception as e:
                    self._log_detalhado(f"Falha com delimitador '{delim}': {e}", "ALERTA")
                    continue
            
            
            # Se nenhum delimitador funcionou, tentar carregamento automático
            self._log_detalhado("Tentando carregamento automático...")
            df = pd.read_csv(csv_path, encoding='utf-8-sig', engine='python')
            self._log_detalhado(f"Carregamento automático: {len(df)} linhas, {len(df.columns)} colunas")
            
            # Se ainda estiver como uma coluna, tentar split manual
            if len(df.columns) == 1:
                self._log_detalhado("Dividindo coluna única manualmente...")
                coluna_unica = df.columns[0]
                # Divide pela vírgula (que parece ser o seu delimitador)
                dados_divididos = df[coluna_unica].str.split(',', expand=True)
                
                # Pega o cabeçalho da primeira linha
                cabecalho = dados_divididos.iloc[0].str.strip()
                dados_divididos = dados_divididos[1:]  # Remove a linha do cabeçalho
                dados_divididos.columns = cabecalho
                
                self._log_detalhado(f"Divisão manual: {len(dados_divididos)} linhas, {len(dados_divididos.columns)} colunas")
                return dados_divididos
            df = self.corrigir_caracteres_especiais_csv(csv_path, df)      
            return df
            
        except pd.errors.ParserError as e:
            self._log_detalhado(f"Erro de parsing detectado: {e}", "ERRO")
            self._log_detalhado("Tentando corrigir automaticamente...")
            
            # Lê o arquivo linha por linha para identificar o problema
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                linhas = f.readlines()
            
            # Identifica a linha problemática
            for i, linha in enumerate(linhas, 1):
                if len(linha.split(',')) != 7:  # Espera 7 colunas
                    self._log_detalhado(f"Linha {i} problemática: {linha.strip()}", "ALERTA")
            
            # Tenta carregar com tratamento de erro mais flexível
            try:
                df = pd.read_csv(csv_path, delimiter=',', encoding='utf-8-sig', 
                            on_bad_lines='skip',  # Pula linhas problemáticas
                            engine='python')
                self._log_detalhado(f"Carregado com {len(df)} linhas após correção automática", "SUCESSO")
                return df
            except:
                # Última tentativa: carrega manualmente
                self._log_detalhado("Carregando manualmente...")
                dados_corrigidos = []
                cabecalho = linhas[0].strip().split(',')
                
                for i, linha in enumerate(linhas[1:], 2):  # Começa da linha 2 (pula cabeçalho)
                    campos = linha.strip().split(',')
                    if len(campos) == len(cabecalho):
                        dados_corrigidos.append(campos)
                    else:
                        self._log_detalhado(f"Linha {i} ignorada: número de campos incorreto ({len(campos)} vs {len(cabecalho)})", "ALERTA")
                
                df = pd.DataFrame(dados_corrigidos, columns=cabecalho)
                self._log_detalhado(f"Carregamento manual: {len(df)} linhas processadas", "SUCESSO")
                return df
                
        except UnicodeDecodeError:
            try:
                return pd.read_csv(csv_path, delimiter=',', encoding='latin-1')
            except:
                return pd.read_csv(csv_path, delimiter=',', encoding='iso-8859-1')
        except Exception as e:
            self._log_detalhado(f"Erro crítico ao carregar dados: {e}", "ERRO")
            return None
    def _calcular_medias_odds_por_tipo(self):
        """Calcular médias de odds por tipo de estatística baseado nos dados históricos"""
        if not hasattr(self, 'df_treino_limpo') or self.df_treino_limpo is None:
            self._log_detalhado("Dados de treino não disponíveis", "ALERTA")
            return {}
        
        try:
            # Garantir que temos as colunas necessárias
            if 'Tipo_Estatistica' not in self.df_treino_limpo.columns or 'Odds' not in self.df_treino_limpo.columns:
                self._log_detalhado("Colunas necessárias não encontradas", "ALERTA")
                return {}
            
            # Filtrar odds válidas (> 1.0)
            df_valido = self.df_treino_limpo[self.df_treino_limpo['Odds'] > 1.0].copy()
            
            if len(df_valido) == 0:
                self._log_detalhado("Nenhuma odds válida encontrada", "ALERTA")
                return {}
            
            # Calcular médias por tipo
            medias = df_valido.groupby('Tipo_Estatistica')['Odds'].agg([
                'mean', 'median', 'count', 'std'
            ]).round(3)
            
            self.medias_odds_por_tipo = {}
            
            self._log_detalhado("📊 MÉDIAS DE ODDS POR TIPO DE ESTATÍSTICA:")
            self._log_detalhado("="*50)
            
            for tipo in medias.index:
                mean_val = medias.loc[tipo, 'mean']
                median_val = medias.loc[tipo, 'median']
                count_val = medias.loc[tipo, 'count']
                std_val = medias.loc[tipo, 'std']
                
                # Usar mediana se tiver outliers, senão média
                if count_val >= 10 and std_val < mean_val * 0.5:  # Pouca variação
                    odds_media = mean_val
                else:
                    odds_media = median_val
                
                self.medias_odds_por_tipo[tipo] = odds_media
                
                self._log_detalhado(f"   {tipo:15s}: Média={mean_val:.2f} | Mediana={median_val:.2f} | "
                                f"N={count_val:3d} | Usando: {odds_media:.2f}")
            
            # Valores padrão se algum tipo não tiver dados
            defaults = {
                'VITORIA': 1.5,
                'DERROTA': 3.0,
                'BTTS': 1.8,
                'OVER_2.5': 1.9,
                'UNDER_2.5': 1.7,
                'EMPATE': 3.5,
                'OUTRO': 2.0
            }
            
            for tipo, default_val in defaults.items():
                if tipo not in self.medias_odds_por_tipo:
                    self.medias_odds_por_tipo[tipo] = default_val
                    self._log_detalhado(f"   {tipo:15s}: Sem dados → usando padrão: {default_val:.2f}", "ALERTA")
            
            return self.medias_odds_por_tipo
            
        except Exception as e:
            self._log_detalhado(f"Erro ao calcular médias de odds: {e}", "ERRO")
            return {}
    def treinar_modelo_evolutivo(self, forcar_retreino=False):
        """Treinar ou atualizar modelo com dados mais recentes - VERSÃO COMPLETA COM MELHORIAS"""
        self._log_detalhado("INICIANDO TREINAMENTO EVOLUTIVO...")
        
        # Verificar se modelo existe e se deve atualizar
        if os.path.exists(self.modelo_path) and not forcar_retreino:
            self._log_detalhado("Modelo existente encontrado. Verificando necessidade de atualização...")
            if not self._verificar_necessidade_atualizacao():
                self._log_detalhado("Modelo atualizado. Carregando modelo existente...", "SUCESSO")
                if self.carregar_modelo():
                    return True
        
        if self.base_treino_path is None:
            self._log_detalhado("Caminho da base de treino não especificado", "ERRO")
            return False
            
        # Carregar e preparar dados
        self._log_detalhado("Carregando dados de treino...")
        self.df_treino = self.carregar_dados(self.base_treino_path)
        
        if self.df_treino is None or len(self.df_treino) == 0:
            self._log_detalhado("Falha ao carregar dados de treino", "ERRO")
            return False
        
        self._log_detalhado(f"Dados carregados: {len(self.df_treino)} registros")
        
        # Validar dados antes de processar
        try:
            self._validar_dados_treino()
        except ValueError as e:
            self._log_detalhado(f"Erro na validação: {e}", "ERRO")
            return False
        
        self._preparar_dados_treino()
        
        # ✅ CORREÇÃO: Verificar se df_treino_limpo foi criado
        if not hasattr(self, 'df_treino_limpo') or len(self.df_treino_limpo) < 10:
            self._log_detalhado("Dados insuficientes para treinamento", "ERRO")
            return False
        
        # ✅ MELHORIA 1: ANALISAR DESEMPENHO POR STREK ANTES DO TREINAMENTO
        self.analisar_desempenho_por_streak(self.df_treino_limpo)
        
        # Treinar novo modelo
        self._log_detalhado("Treinando novo modelo com dados atualizados...")
        try:
            # ✅ CORREÇÃO: Processar colunas categóricas ANTES de criar X
            df_processado = self.df_treino_limpo.copy()
            
            # ✅ MELHORIA 2: ADICIONAR NOVAS FEATURES DE STREAK
            # FEATURE CRÍTICA: Tamanho da streak ao quadrado (streaks longos não são lineares)
            df_processado['Streak_Squared'] = df_processado['Tamanho_Streak'] ** 2
            
            # FEATURE CRÍTICA: Interação entre streak e tipo de estatística
            tipo_encoded = pd.factorize(df_processado['Tipo_Estatistica'])[0]
            df_processado['Streak_Tipo_Interaction'] = df_processado['Tamanho_Streak'] * tipo_encoded
            
            # FEATURE CRÍTICA: Streak muito longo (>10 jogos)
            df_processado['Streak_Longo'] = (df_processado['Tamanho_Streak'] > 10).astype(int)
            
            # FEATURE CRÍTICA: Streak muito curto (<=3 jogos)
            df_processado['Streak_Curto'] = (df_processado['Tamanho_Streak'] <= 3).astype(int)
            
            # FEATURE CRÍTICA: Streak médio (4-8 jogos)
            df_processado['Streak_Medio'] = ((df_processado['Tamanho_Streak'] >= 4) & 
                                            (df_processado['Tamanho_Streak'] <= 8)).astype(int)
            
            self._log_detalhado("✅ Novas features de streak adicionadas:", "SUCESSO")
            self._log_detalhado(f"   - Streak_Squared: {df_processado['Streak_Squared'].mean():.2f}")
            self._log_detalhado(f"   - Streak_Tipo_Interaction: {df_processado['Streak_Tipo_Interaction'].mean():.2f}")
            self._log_detalhado(f"   - Streak_Longo (>10): {df_processado['Streak_Longo'].mean():.1%}")
            self._log_detalhado(f"   - Streak_Medio (4-8): {df_processado['Streak_Medio'].mean():.1%}")
            
            # 1. Codificar colunas categóricas
            # Para 'Tipo_Estatistica'
            if 'Tipo_Estatistica' in df_processado.columns:
                unique_stats = df_processado['Tipo_Estatistica'].unique()
                self._log_detalhado(f"Tipos de estatística encontrados: {list(unique_stats)}")
                # Criar mapeamento manual
                self.stat_mapping = {stat: idx for idx, stat in enumerate(unique_stats)}
                df_processado['Tipo_Estatistica_encoded'] = df_processado['Tipo_Estatistica'].map(self.stat_mapping)
            
            # Para 'Local_Jogo'
            if 'Local_Jogo' in df_processado.columns:
                local_encoder = LabelEncoder()
                df_processado['Local_Jogo_encoded'] = local_encoder.fit_transform(df_processado['Local_Jogo'])
            
            # Para 'Liga_Categoria'
            if 'Liga_Categoria' in df_processado.columns:
                liga_encoder = LabelEncoder()
                df_processado['Liga_Categoria_encoded'] = liga_encoder.fit_transform(df_processado['Liga_Categoria'])
            
            # ✅ CORREÇÃO IMPORTANTE: Para 'Streak_Categoria' 
            if 'Streak_Categoria' in df_processado.columns:
                # Primeiro, converter de Categorical para string se necessário
                if pd.api.types.is_categorical_dtype(df_processado['Streak_Categoria']):
                    df_processado['Streak_Categoria'] = df_processado['Streak_Categoria'].astype(str)
                
                # Verificar valores únicos
                unique_streaks = df_processado['Streak_Categoria'].dropna().unique()
                self._log_detalhado(f"Valores de Streak_Categoria: {list(unique_streaks)}")
                
                # Opção mais simples: usar mapeamento manual (evita problemas com Categorical)
                streak_order = {'Curto': 0, 'Medio': 1, 'Longo': 2, 'Muito_Longo': 3, 'nan': -1}
                
                # Converter NaN para string 'nan' temporariamente para mapeamento
                streak_series = df_processado['Streak_Categoria'].fillna('nan').astype(str)
                df_processado['Streak_Categoria_encoded'] = streak_series.map(
                    lambda x: streak_order.get(x, -1)
                )
            
            # ✅ CORREÇÃO: Para 'Dia_Semana' - garantir que é numérico
            if 'Dia_Semana' in df_processado.columns:
                # Se for float com NaN, converter para int (preenchendo NaN com -1)
                if df_processado['Dia_Semana'].dtype == 'float64':
                    df_processado['Dia_Semana'] = df_processado['Dia_Semana'].fillna(-1).astype(int)
            
            # ✅ MELHORIA 3: ADICIONAR FEATURE DE PERFORMANCE POR TIPO DE ESTATÍSTICA
            # Calcular taxa de acerto por tipo de estatística
            if 'Tipo_Estatistica' in df_processado.columns and 'Target' in df_processado.columns:
                tipo_performance = df_processado.groupby('Tipo_Estatistica')['Target'].mean().to_dict()
                self._log_detalhado("Performance por tipo de estatística:", "INFO")
                for tipo, taxa in tipo_performance.items():
                    self._log_detalhado(f"   {tipo}: {taxa:.1%} acertos")
                
                # Adicionar como feature
                df_processado['Tipo_Performance'] = df_processado['Tipo_Estatistica'].map(tipo_performance)
            
            # ✅ CORREÇÃO: Definir features numéricas que vamos usar
            features_numericas = [
                'Odds', 'Tamanho_Streak', 'Odds_Streak_Interaction', 'Prob_Implicita',
                'Streak_Squared', 'Streak_Tipo_Interaction', 'Streak_Longo', 
                'Streak_Curto', 'Streak_Medio'
            ]
            
            # Adicionar performance por tipo se existir
            if 'Tipo_Performance' in df_processado.columns:
                features_numericas.append('Tipo_Performance')
            
            # Adicionar features encoded se existirem
            encoded_features = []
            if 'Tipo_Estatistica_encoded' in df_processado.columns:
                encoded_features.append('Tipo_Estatistica_encoded')
            if 'Local_Jogo_encoded' in df_processado.columns:
                encoded_features.append('Local_Jogo_encoded')
            if 'Liga_Categoria_encoded' in df_processado.columns:
                encoded_features.append('Liga_Categoria_encoded')
            if 'Streak_Categoria_encoded' in df_processado.columns:
                encoded_features.append('Streak_Categoria_encoded')
            if 'Dia_Semana' in df_processado.columns:
                encoded_features.append('Dia_Semana')
            
            # Combinar todas as features
            todas_features = features_numericas + encoded_features
            
            # Verificar se todas as features existem
            features_existentes = [f for f in todas_features if f in df_processado.columns]
            self._log_detalhado(f"Features a usar: {features_existentes}")
            
            # ✅ CORREÇÃO: Garantir que todas as features são numéricas e sem NaN
            for feature in features_existentes:
                if feature in df_processado.columns:
                    # Converter para numérico, forçando erros para NaN
                    df_processado[feature] = pd.to_numeric(df_processado[feature], errors='coerce')
            
            # Criar X apenas com features existentes
            X = df_processado[features_existentes].copy()
            y = df_processado['Target']
            
            # Remover possíveis valores NaN
            mask = X.notna().all(axis=1)
            X = X[mask].copy()
            y = y[mask].copy()
            
            if len(X) == 0:
                self._log_detalhado("❌ Nenhuma feature válida após limpeza", "ERRO")
                return False
            
            # ✅ MELHORIA 4: ANALISAR CORRELAÇÃO DAS FEATURES COM O TARGET
            self._analisar_correlacao_features(X, y)
            
            self._log_detalhado(f"Shape final: X={X.shape}, y={y.shape}")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # ✅ CORREÇÃO: Balanceamento de classes
            from sklearn.utils import class_weight
            classes = np.unique(y_train)
            pesos = class_weight.compute_class_weight(
                'balanced', 
                classes=classes, 
                y=y_train
            )
            class_weights = dict(zip(classes, pesos))
            
            # ✅ MELHORIA 5: PARÂMETROS OTIMIZADOS PARA STREAKS
            self.model = RandomForestClassifier(
                n_estimators=500,  # Aumentado para melhor capturar padrões de streak
                max_depth=20,      # Aumentado para capturar relações não-lineares
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                class_weight=class_weights,
                random_state=42,
                n_jobs=-1,
                bootstrap=True,
                oob_score=True
            )
            
            # Treinar modelo
            self._log_detalhado("Treinando modelo...")
            self.model.fit(X_train, y_train)
            
            # ✅ MELHORIA 6: AVALIAÇÃO MAIS DETALHADA
            # Avaliação OOB (out-of-bag)
            if hasattr(self.model, 'oob_score_'):
                self._log_detalhado(f"OOB Score: {self.model.oob_score_:.3f}", "INFO")
            
            # Avaliação
            accuracy = self.model.score(X_test, y_test)
            self.acuracia_modelo = accuracy
            
            # Avaliação detalhada
            self._avaliar_modelo_detalhado(X_test, y_test)
            
            # ✅ MELHORIA 7: ANÁLISE DE IMPORTÂNCIA DAS FEATURES
            self._analisar_importancia_features(X.columns)
            
            # Salvar informações adicionais sobre o encoding
            self.features_para_treino = features_existentes
            self.encoders = {
                'local_encoder': local_encoder if 'Local_Jogo' in df_processado.columns else None,
                'liga_encoder': liga_encoder if 'Liga_Categoria' in df_processado.columns else None,
                'stat_mapping': self.stat_mapping if 'Tipo_Estatistica' in df_processado.columns else None,
                'streak_order': streak_order if 'Streak_Categoria' in df_processado.columns else None,
                'tipo_performance': tipo_performance if 'Tipo_Performance' in df_processado.columns else None
            }
            
            # ✅ MODIFICAÇÃO: Fazer backup apenas se acurácia melhorar
            backup_feito = self._fazer_backup_modelo(self.acuracia_modelo)
            
            # Salvar modelo com todas as informações
            self._salvar_modelo_evolutivo()
            
            self._log_detalhado(f"Modelo treinado com sucesso! Acurácia: {accuracy:.2%}", "SUCESSO")
            if backup_feito:
                self._log_detalhado("✅ Backup automático criado (acurácia melhorou)", "SUCESSO")
            
            self._log_detalhado(f"Total de amostras de treino: {len(self.df_treino_limpo)}")
            self._log_detalhado(f"Número de features: {len(features_existentes)}")
            
            return True
            
        except Exception as e:
            self._log_detalhado(f"Erro no treinamento: {e}", "ERRO")
            import traceback
            self._log_detalhado(f"Detalhes do erro: {traceback.format_exc()}", "ERRO")
            return False
    def analisar_desempenho_por_streak(self, df):
        """Analisar performance real por tamanho de streak - para debug"""
        if not hasattr(self, 'df_treino_limpo'):
            return
        
        # Agrupar por tamanho da streak
        stats_streak = self.df_treino_limpo.groupby('Tamanho_Streak').agg({
            'Target': ['count', 'mean', 'std'],
            'Odds': 'mean'
        }).round(3)
        
        self._log_detalhado("📊 DESEMPENHO REAL POR TAMANHO DE STREAK:")
        for streak in sorted(stats_streak.index):
            dados = stats_streak.loc[streak]
            total = dados[('Target', 'count')]
            taxa = dados[('Target', 'mean')]
            odds_media = dados[('Odds', 'mean')]
            
            self._log_detalhado(f"   Streak {streak}: {total} jogos, {taxa:.1%} acertos, odds média {odds_media:.2f}")
    def _salvar_modelo_evolutivo(self):
        """Salvar modelo evolutivo com todas as informações necessárias"""
        try:
            # Coletar todas as informações dos encoders
            encoders_info = {}
            
            # Salvar mapeamentos para cada encoder usado
            if hasattr(self, 'encoders'):
                for encoder_name, encoder in self.encoders.items():
                    if encoder is not None:
                        if hasattr(encoder, 'classes_'):
                            encoders_info[encoder_name] = {
                                'type': 'LabelEncoder',
                                'classes': list(encoder.classes_)
                            }
                        elif isinstance(encoder, dict):
                            encoders_info[encoder_name] = {
                                'type': 'dict_mapping',
                                'mapping': encoder
                            }
            
            modelo_salvo = {
                'model': self.model,
                'features': self.features_para_treino,
                'acuracia': self.acuracia_modelo,
                'encoders_info': encoders_info,
                'stat_mapping': self.stat_mapping if hasattr(self, 'stat_mapping') else {},
                'n_amostras': len(self.df_treino_limpo) if hasattr(self, 'df_treino_limpo') else 0,
                'timestamp': pd.Timestamp.now()
            }
            
            joblib.dump(modelo_salvo, self.modelo_path)
            self._log_detalhado(f"Modelo salvo em: {self.modelo_path}", "SUCESSO")
            return True
        except Exception as e:
            self._log_detalhado(f"Erro ao salvar modelo: {e}", "ERRO")
            return False   
    def _criar_modelo_ensemble(self):
        """Criar ensemble para melhor performance"""
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        models = [
            ('rf', RandomForestClassifier(n_estimators=300, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('svc', SVC(probability=True, random_state=42))
        ]
        
        return VotingClassifier(estimators=models, voting='soft')
    def _otimizar_hyperparametros(self, X_train, y_train):
        """Otimizar hyperparâmetros automaticamente"""
        from sklearn.model_selection import GridSearchCV
        
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 2, 3]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self._log_detalhado(f"✅ Melhores parâmetros: {grid_search.best_params_}")
        self._log_detalhado(f"✅ Melhor score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_
    """ def criar_backup_manual(self): 
        Criar backup manual do modelo atual SEM verificar acurácia
        if not os.path.exists(self.modelo_path):
            self._log_detalhado("Nenhum modelo encontrado para backup", "ALERTA")
            return False
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.modelo_path}.backup_manual_{timestamp}"
            import shutil
            shutil.copy2(self.modelo_path, backup_path)
            self._log_detalhado(f"✅ Backup MANUAL criado: {backup_path}", "SUCESSO")
            return True
        except Exception as e:
            self._log_detalhado(f"❌ Erro ao criar backup manual: {e}", "ERRO")
            return False   """
    def _avaliar_modelo_detalhado(self, X_test, y_test):
        """Avaliação mais detalhada do modelo"""
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.model_selection import cross_val_score
        # ✅ ADICIONAR validação cruzada
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5, scoring='accuracy')
        
        self._log_detalhado("VALIDAÇÃO CRUZADA (5-fold):")
        self._log_detalhado(f"  Acurácia média: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        self._log_detalhado(f"  Scores individuais: {cv_scores}")
        
        y_pred = self.model.predict(X_test)
        
        self._log_detalhado("AVALIAÇÃO DETALHADA DO MODELO:")
        self._log_detalhado("="*40)
        self._log_detalhado(f"Acurácia: {self.acuracia_modelo:.2%}")
        self._log_detalhado("Matriz de Confusão:")
        self._log_detalhado(str(confusion_matrix(y_test, y_pred)))
        self._log_detalhado("Relatório de Classificação:")
        self._log_detalhado(classification_report(y_test, y_pred))

    def _verificar_necessidade_atualizacao(self):
        """Verificar se base de treino tem novos dados SEM fazer backup automático"""
        if not os.path.exists(self.modelo_path) or self.base_treino_path is None:
            return True
            
        # Carregar dados atuais
        df_atual = self.carregar_dados(self.base_treino_path)
        
        if df_atual is None or len(df_atual) == 0:
            self._log_detalhado("Falha ao carregar dados para verificação", "ERRO")
            return True
        
        # CORREÇÃO: Verificar se a coluna existe após carregamento correto
        if 'Situação' not in df_atual.columns:
            self._log_detalhado("Coluna 'Situação' não encontrada após carregamento", "ERRO")
            self._log_detalhado(f"Colunas disponíveis: {list(df_atual.columns)}")
            return True
        
        # Filtrar dados válidos
        df_atual_limpo = df_atual[df_atual['Situação'].notna() & 
                                (df_atual['Situação'] != '')]
        
        # Carregar info do modelo salvo
        try:
            modelo_data = joblib.load(self.modelo_path)
            amostras_anteriores = modelo_data.get('amostras_treino', 0)
            
            self._log_detalhado(f"Comparação: Modelo atual {amostras_anteriores} vs Dados {len(df_atual_limpo)}")
            
            # Se tem pelo menos 10% mais dados, retreinar
            if len(df_atual_limpo) > amostras_anteriores * 1.1:
                self._log_detalhado(f"📈 Novos dados detectados: {amostras_anteriores} → {len(df_atual_limpo)}", "SUCESSO")
                # ✅ REMOVIDO: Não faz mais backup automático aqui
                return True
            else:
                self._log_detalhado(f"📊 Modelo já está atualizado", "SUCESSO")
                
        except Exception as e:
            self._log_detalhado(f"Erro ao verificar modelo: {e}", "ALERTA")
            return True
            
        return False

    
    def _classificar_estatistica(self, stat):
        """🔥 CLASSIFICAÇÃO DINÂMICA DE ESTATÍSTICAS"""
        stat_str = str(stat).lower()
        
        # Padrões dinâmicos baseados em performance real
        padroes = {
            'VITORIA': ['won', 'win', 'victory', 'wins', 'winning'],
            'DERROTA': ['lost', 'loss', 'defeat', 'losing'], 
            'EMPATE': ['drew', 'draw', 'tie', 'drawing'],
            'BTTS': ['btts', 'both teams score', 'ambos marcam'],
            'OVER_2.5': ['over 2.5', 'over 3.5', '+2.5', '+3.5'],
            'UNDER_2.5': ['under 2.5', 'under 3.5', '-2.5', '-3.5']
        }
        
        for tipo, palavras in padroes.items():
            if any(palavra in stat_str for palavra in palavras):
                return tipo
        
        return 'OUTRO'
    def _extrair_streak(self, stat):
        """Extrair tamanho do streak com múltiplos padrões - VERSÃO MELHORADA"""
        stat_str = str(stat).lower()
        
        # Múltiplos padrões de extração
        padroes = [
            r'last (\d+)',           # "last 6"
            r'(\d+) consecutive',    # "6 consecutive"
            r'(\d+) straight',       # "6 straight"
            r'(\d+) in a row',       # "6 in a row"
            r'(\d+) games',          # "6 games"
            r'(\d+) matches',        # "6 matches"
            r'(\d+) league matches', # "6 league matches"
            r'(\d+) home matches',   # "6 home matches"
            r'(\d+) away matches',   # "6 away matches"
        ]
        
        for padrao in padroes:
            matches = re.findall(padrao, stat_str)
            if matches:
                try:
                    streak = int(matches[0])
                    # DEBUG: Log para verificar extração
                    self._log_detalhado(f"🔍 Streak extraído: {streak} do padrão: {padrao}", "INFO")
                    return streak
                except:
                    continue
        
        # Fallback: procurar por qualquer número no início do padrão
        match_fallback = re.search(r'(\d+)\s+(?:league\s+)?matches', stat_str)
        if match_fallback:
            try:
                streak = int(match_fallback.group(1))
                self._log_detalhado(f"🔍 Streak fallback: {streak}", "INFO")
                return streak
            except:
                pass
        
        self._log_detalhado(f"⚠️  Streak não encontrado em: {stat_str[:50]}", "ALERTA")
        return 6  # Valor padrão mais comum na sua base
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
        """🔥 CLASSIFICAÇÃO DINÂMICA BASEADA EM DESEMPENHO REAL"""
        if pd.isna(league) or league == '':
            return 'MEDIA_CONFIABILIDADE'
        
        league_str = str(league).strip()
        
        # Se temos dados de treino, calcular confiabilidade baseada em desempenho real
        if hasattr(self, 'df_treino_limpo'):
            return self._calcular_confiabilidade_por_desempenho(league_str)
        else:
            # Fallback para primeira execução
            return self._classificar_liga_fallback(league_str)
    def _calcular_confiabilidade_por_desempenho(self, league_str):
            """
            Classificar liga baseada em desempenho histórico dos times - VERSÃO SIMPLIFICADA
            """
            try:
                # Fallback para classificação básica se não temos dados suficientes
                if not hasattr(self, 'df_treino_limpo') or self.df_treino_limpo is None:
                    return self._classificar_liga_fallback(league_str)
                
                # Verificar quantos jogos temos para esta liga
                jogos_liga = self.df_treino_limpo[self.df_treino_limpo['League'] == league_str]
                
                if len(jogos_liga) < 10:  # Poucos dados
                    return 'MEDIA_CONFIABILIDADE'
                
                # Calcular taxa de acerto para esta liga
                taxa_acerto = jogos_liga['Target'].mean()
                
                # Classificar baseado na taxa de acerto
                if taxa_acerto >= 0.60:
                    return 'ALTA_CONFIABILIDADE'
                elif taxa_acerto >= 0.50:
                    return 'MEDIA_CONFIABILIDADE'
                elif taxa_acerto >= 0.40:
                    return 'BAIXA_CONFIABILIDADE'
                else:
                    return 'MUITO_BAIXA_CONFIABILIDADE'
                    
            except Exception as e:
                self._log_detalhado(f"Erro em calcular_confiabilidade: {e}", "ALERTA")
                return 'MEDIA_CONFIABILIDADE'
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
        """Classificar padrão baseado na probabilidade - VERSÃO CORRIGIDA"""
        if probabilidade > 0.75:
            return 'PADRAO_FORTISSIMO'
        elif probabilidade > 0.68:
            return 'PADRAO_FORTE'
        elif probabilidade > 0.60:
            return 'PADRAO_SOLIDO'
        elif probabilidade > 0.55:
            return 'PADRAO_REGULAR'
        else:
            return 'PADRAO_FRACO'

    def _classificar_recomendacao(self, probabilidade):
        """Classificar recomendação - VERSÃO CORRIGIDA"""
        if probabilidade > 0.75:
            return 'EXCELENTE'
        elif probabilidade > 0.65:
            return 'BOA'
        elif probabilidade > 0.55:
            return 'REGULAR'
        else:
            return 'ARISCADO'
    def _gerar_analise_detalhada(self, prob, previsao, tipo_estatistica, streak, odds, padrao, bonus, liga):
        """Gerar análise detalhada"""
        base = f"{padrao}: {tipo_estatistica} em {streak} jogos (Odds: {odds})"
        if bonus > 0:
            base += f" [BONUS +{bonus:.0%}]"
        base += f" Previsao: {'MANTERA STREAK' if previsao == 1 else 'INTERROMPERA STREAK'}"
        return base

    def _converter_data_ingles_para_brasil(self, date_str):
        """Converter formato 'Sunday, 13 October 21:30' para '13/10/2024 21:30'"""
        try:
            meses_ingles = {
                'january': '01', 'february': '02', 'march': '03', 'april': '04',
                'may': '05', 'june': '06', 'july': '07', 'august': '08',
                'september': '09', 'october': '10', 'november': '11', 'december': '12'
            }
            
            # Remove dia da semana se existir
            partes = date_str.strip().split(',')
            if len(partes) > 1:
                # Formato: "Sunday, 13 October 21:30"
                data_hora = partes[1].strip()
            else:
                # Formato: "13 October 21:30"
                data_hora = partes[0].strip()
            
            # Divide data e hora
            partes_data_hora = data_hora.split()
            if len(partes_data_hora) >= 3:
                dia = partes_data_hora[0].zfill(2)
                mes_ingles = partes_data_hora[1].lower()
                hora = partes_data_hora[2]
                
                mes_numero = meses_ingles.get(mes_ingles, '01')
                ano_atual = datetime.now().year
                
                return f"{dia}/{mes_numero}/{ano_atual} {hora}"
            
            return date_str
        except Exception as e:
            self._log_detalhado(f"Erro ao converter data '{date_str}': {e}", "ALERTA")
            return date_str

    def _converter_data_ingles_simples_para_brasil(self, date_str):
        """Converter formato '13 October 21:30' para '13/10/2024 21:30'"""
        try:
            meses_ingles = {
                'january': '01', 'february': '02', 'march': '03', 'april': '04',
                'may': '05', 'june': '06', 'july': '07', 'august': '08',
                'september': '09', 'october': '10', 'november': '11', 'december': '12'
            }
            
            partes = date_str.strip().split()
            if len(partes) >= 3:
                dia = partes[0].zfill(2)
                mes_ingles = partes[1].lower()
                hora = partes[2]
                
                mes_numero = meses_ingles.get(mes_ingles, '01')
                ano_atual = datetime.now().year
                
                return f"{dia}/{mes_numero}/{ano_atual} {hora}"
            
            return date_str
        except Exception as e:
            self._log_detalhado(f"Erro ao converter data simples '{date_str}': {e}", "ALERTA")
            return date_str

    def _preparar_dados_futuros(self, df_futuros):
        """Preparar dados futuros"""
        self._log_detalhado("Preparando dados futuros...")
        
        # Processar Date - converter do formato inglês
        # if 'Resultado' not in df_futuros.columns:
        #     df_futuros.insert(df_futuros.columns.get_loc('Situação'), 'Resultado', df_futuros['Situação'])
        
            

        if 'Date' in df_futuros.columns:
            df_futuros['Date'] = df_futuros['Date'].apply(self._formatar_data_saida)
            self._log_detalhado(f"Datas processadas: {df_futuros['Date'].notna().sum()} válidas")
        else:
            self._log_detalhado("Coluna 'Date' não encontrada", "ALERTA")
            df_futuros['Date'] = "DATA_INDISPONIVEL"
        
        # Processar Odds - converter para numérico
        if 'Odds' in df_futuros.columns:
            df_futuros['Odds'] = pd.to_numeric(df_futuros['Odds'], errors='coerce')
            self._log_detalhado(f"Odds processadas: {df_futuros['Odds'].notna().sum()} válidas")
        else:
            self._log_detalhado("Coluna 'Odds' não encontrada", "ALERTA")
            df_futuros['Odds'] = np.nan
        
        # ✅ CORREÇÃO: REMOVER INDENTAÇÃO EXTRA AQUI
        # Processar Stat - extrair informações
        if 'Stat' in df_futuros.columns:
            df_futuros['Tipo_Estatistica'] = df_futuros['Stat'].apply(self._classificar_estatistica)
            df_futuros['Tamanho_Streak'] = df_futuros['Stat'].apply(self._extrair_streak)
            
            # EXTRAIR TIME PRINCIPAL CORRETAMENTE - VERSÃO MELHORADA
            def extrair_time_melhorado(stat):
                stat_str = str(stat)
                time = self._extrair_time_avancado(stat_str)
                
                # DEBUG: Logar casos problemáticos
                if time == 'TIME_DESCONHECIDO':
                    self._log_detalhado(f"⚠️  Time não extraído - Stat: '{stat_str[:80]}...'", "ALERTA")
                else:
                    self._log_detalhado(f"✅ Time extraído: '{time}'", "SUCESSO")
                    
                return time
            
            df_futuros['Time'] = df_futuros['Stat'].apply(extrair_time_melhorado)
            
            # Contar estatísticas de extração
            times_unicos = df_futuros['Time'].nunique()
            times_desconhecidos = (df_futuros['Time'] == 'TIME_DESCONHECIDO').sum()
            
            self._log_detalhado(f"📊 Estatísticas de extração: {times_unicos} times únicos, {times_desconhecidos} desconhecidos")
            
            # DEBUG: Ver alguns times extraídos
            self._log_detalhado("DEBUG - Primeiros 5 times extraídos:")
            for i, (stat, time) in enumerate(zip(df_futuros['Stat'].head(), df_futuros['Time'].head())):
                self._log_detalhado(f"   {i+1}. Stat: '{stat}' → Time: '{time}'")
        else:
            self._log_detalhado("Coluna 'Stat' não encontrada - ESSENCIAL", "ERRO")
            df_futuros['Tipo_Estatistica'] = 'OUTRO'
            df_futuros['Tamanho_Streak'] = 1
            df_futuros['Time'] = 'TIME_DESCONHECIDO'
        
        # Processar Next Match - extrair local e formatar
        if 'Next Match' in df_futuros.columns:
            df_futuros['Local_Jogo'] = df_futuros['Next Match'].apply(self._extrair_local)
            # Aplicar formatação diretamente na coluna Next Match existente
            df_futuros['Next Match'] = df_futuros.apply(self._formatar_next_match, axis=1)
            self._log_detalhado("Next Match processado e formatado")
        else:
            self._log_detalhado("Coluna 'Next Match' não encontrada", "ALERTA")
            df_futuros['Local_Jogo'] = 'NEUTRO'
        
        # Processar League - classificar categoria
        if 'League' in df_futuros.columns:
            df_futuros['Liga_Categoria'] = df_futuros['League'].apply(self._classificar_liga)
            self._log_detalhado("Liga categorizada")
        else:
            self._log_detalhado("Coluna 'League' não encontrada", "ALERTA")
            df_futuros['Liga_Categoria'] = 'MEDIA_CONFIABILIDADE'
        
        # Adicionar coluna Situação vazia para consistência
        if 'Situação' not in df_futuros.columns:
            df_futuros['Situação'] = ''
            self._log_detalhado("Coluna Situação adicionada (vazia para jogos futuros)")
        
        self._log_detalhado(f"Dados futuros preparados: {len(df_futuros)} registros")
        self._log_detalhado(f"Colunas disponíveis: {list(df_futuros.columns)}")
        
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
            self._log_detalhado(f"Next Match formatado: '{next_match}' -> '{resultado}'")
            return resultado
        elif 'away' in next_match_lower:
            resultado = next_match_lower.replace('away', time_principal.upper())
            self._log_detalhado(f"Next Match formatado: '{next_match}' -> '{resultado}'")
            return resultado
        else:
            return next_match

    def _formatar_data_saida(self, date_val):
        """Formatar data para saída no formato dd/mm/aaaa hh:mm - VERSÃO CORRIGIDA"""
        if pd.isna(date_val) or date_val == '' or date_val == 'DATA_INDISPONIVEL':
            return "DATA_INDISPONIVEL"
        
        try:
            # Se já for datetime, formatar diretamente
            if isinstance(date_val, (pd.Timestamp, datetime)):
                return date_val.strftime('%d/%m/%Y %H:%M')
            
            # Se for string, verificar diferentes formatos
            date_str = str(date_val).strip()
            
            # FORMATO 1: Já está no formato desejado "20/10/2025 14:10"
            if re.match(r'\d{2}/\d{2}/\d{4} \d{1,2}:\d{2}', date_str):
                # Já está no formato correto, apenas garantir que está padronizado
                try:
                    # Converter para datetime e depois formatar para garantir padrão
                    dt = datetime.strptime(date_str, '%d/%m/%Y %H:%M')
                    return dt.strftime('%d/%m/%Y %H:%M')
                except:
                    return date_str  # Retornar original se não conseguir converter
            
            # FORMATO 2: Formato inglês "Sunday, 19 October 18:00"
            if ',' in date_str and any(day in date_str.lower() for day in 
                ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
                return self._converter_data_ingles_para_brasil(date_str)
            
            # FORMATO 3: "13 October 21:30" (sem dia da semana)
            if any(mes in date_str.lower() for mes in 
                ['january', 'february', 'march', 'april', 'may', 'june', 
                'july', 'august', 'september', 'october', 'november', 'december']):
                return self._converter_data_ingles_simples_para_brasil(date_str)
            
            # Tentar outros formatos conhecidos
            formatos_tentativos = [
                '%Y-%m-%d %H:%M:%S', 
                '%d/%m/%Y %H:%M', 
                '%m/%d/%Y %H:%M', 
                '%d-%m-%Y %H:%M',
                '%d/%m/%Y %H:%M:%S',
                '%Y-%m-%d %H:%M'
            ]
            
            for fmt in formatos_tentativos:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime('%d/%m/%Y %H:%M')
                except:
                    continue
            
            self._log_detalhado(f"Formato de data não reconhecido: {date_str}", "ALERTA")
            return date_str  # Retornar original se não conseguir converter
                
        except Exception as e:
            self._log_detalhado(f"Erro ao formatar data '{date_val}': {e}", "ALERTA")
            return str(date_val)

    def _analise_basica_futuro(self, row):
        
        """Análise básica melhorada com probabilidade variável"""
        try:
            # Calcular probabilidade baseada nas odds
            odds = row.get('Odds', 2.0)
            prob = self._calcular_probabilidade_odds(odds)
            
            # Ajustar baseado no tipo de estatística
            tipo_estatistica = self._classificar_estatistica(row.get('Stat', ''))
            if tipo_estatistica in ['VITORIA', 'BTTS']:
                prob *= 1.05  # Leve aumento para tipos mais confiáveis
            elif tipo_estatistica in ['DERROTA', 'EMPATE']:
                prob *= 0.95  # Leve redução para tipos menos confiáveis
            
            # Garantir faixa razoável
            prob = max(0.45, min(0.75, prob))
            
            # Determinar previsão baseada na probabilidade
            previsao = 'VERDADEIRO' if prob > 0.55 else 'FALSO'
            
            return {
                'Probabilidade_Sucesso': round(prob, 3),
                'Previsao': previsao,
                'Padrao': 'PADRAO_REGULAR',
                'Recomendacao': 'REGULAR',
                'Analise_Detalhada': 'Análise básica com odds consideradas',
                'Bonus_Total': 0.0
            }
        except:
            # Fallback mínimo
            return {
                'Probabilidade_Sucesso': 0.6,
                'Previsao': 'VERDADEIRO',
                'Padrao': 'PADRAO_REGULAR',
                'Recomendacao': 'REGULAR',
                'Analise_Detalhada': 'Análise básica - dados insuficientes',
                'Bonus_Total': 0.0
            }
    
    def carregar_modelo(self):
        """Carregar modelo com todas as informações e recriar encoders"""
        try:
            if not os.path.exists(self.modelo_path):
                self._log_detalhado(f"Modelo não encontrado: {self.modelo_path}", "ERRO")
                return False
            
            modelo_salvo = joblib.load(self.modelo_path)
            
            # Carregar componentes principais
            self.model = modelo_salvo['model']
            self.features_para_treino = modelo_salvo['features']
            self.acuracia_modelo = modelo_salvo['acuracia']
            
            # Recriar encoders a partir das informações salvas
            if 'encoders_info' in modelo_salvo:
                self.encoders = {}
                for encoder_name, encoder_info in modelo_salvo['encoders_info'].items():
                    if encoder_info['type'] == 'LabelEncoder':
                        encoder = LabelEncoder()
                        encoder.classes_ = np.array(encoder_info['classes'])
                        self.encoders[encoder_name] = encoder
                    elif encoder_info['type'] == 'dict_mapping':
                        self.encoders[encoder_name] = encoder_info['mapping']
            
            # Carregar mapeamento de estatísticas
            if 'stat_mapping' in modelo_salvo:
                self.stat_mapping = modelo_salvo['stat_mapping']
            
            self._log_detalhado(f"Modelo carregado - Acurácia: {self.acuracia_modelo:.2%}", "SUCESSO")
            self._log_detalhado(f"Features: {self.features_para_treino}")
            return True
            
        except Exception as e:
            self._log_detalhado(f"Erro ao carregar modelo: {e}", "ERRO")
            return False
    def _analisar_jogo_individual(self, jogo_features):
        """Analisar um jogo individual com o modelo - VERSÃO CORRIGIDA"""
        try:
            # Verificar se temos o modelo carregado
            if not hasattr(self, 'model') or self.model is None:
                self._log_detalhado("Modelo não carregado", "ERRO")
                return None
            
            # Criar dataframe com features do jogo
            jogo_df = pd.DataFrame([jogo_features])
            
            # ✅ CORREÇÃO: Processar features categóricas usando os encoders salvos
            # Processar Tipo_Estatistica
            if 'Tipo_Estatistica' in jogo_df.columns and 'stat_mapping' in self.encoders:
                estatistica = jogo_df['Tipo_Estatistica'].iloc[0]
                # Usar valor padrão se não estiver no mapeamento
                jogo_df['Tipo_Estatistica_encoded'] = self.encoders['stat_mapping'].get(estatistica, -1)
            
            # Processar Local_Jogo
            if 'Local_Jogo' in jogo_df.columns and 'local_encoder' in self.encoders:
                local = str(jogo_df['Local_Jogo'].iloc[0])
                encoder = self.encoders['local_encoder']
                # Verificar se o valor está nas classes do encoder
                if local in encoder.classes_:
                    jogo_df['Local_Jogo_encoded'] = encoder.transform([local])[0]
                else:
                    jogo_df['Local_Jogo_encoded'] = -1  # Valor padrão
            
            # Processar Liga_Categoria
            if 'Liga_Categoria' in jogo_df.columns and 'liga_encoder' in self.encoders:
                liga = str(jogo_df['Liga_Categoria'].iloc[0])
                encoder = self.encoders['liga_encoder']
                if liga in encoder.classes_:
                    jogo_df['Liga_Categoria_encoded'] = encoder.transform([liga])[0]
                else:
                    jogo_df['Liga_Categoria_encoded'] = -1
            
            # Processar Streak_Categoria
            if 'Streak_Categoria' in jogo_df.columns and 'streak_order' in self.encoders:
                streak = str(jogo_df['Streak_Categoria'].iloc[0])
                jogo_df['Streak_Categoria_encoded'] = self.encoders['streak_order'].get(streak, -1)
            
            # ✅ CORREÇÃO: Garantir que temos todas as features necessárias
            features_necessarias = self.features_para_treino
            
            # Criar dataframe com todas as features necessárias
            features_finais = {}
            for feature in features_necessarias:
                if feature in jogo_df.columns:
                    features_finais[feature] = jogo_df[feature].iloc[0]
                else:
                    # Valor padrão para features faltantes
                    if feature.endswith('_encoded'):
                        features_finais[feature] = -1
                    elif feature == 'Odds':
                        features_finais[feature] = 2.0  # Valor padrão para odds
                    elif feature == 'Tamanho_Streak':
                        features_finais[feature] = 5.0  # Valor padrão
                    elif feature == 'Odds_Streak_Interaction':
                        features_finais[feature] = 10.0  # Valor padrão
                    elif feature == 'Prob_Implicita':
                        features_finais[feature] = 0.5  # Valor padrão
                    elif feature == 'Dia_Semana':
                        features_finais[feature] = -1
                    else:
                        features_finais[feature] = 0.0
            
            # Criar dataframe final para predição
            X_pred = pd.DataFrame([features_finais])[features_necessarias]
            
            # Fazer predição
            proba = self.model.predict_proba(X_pred)[0]
            predicao = self.model.predict(X_pred)[0]
            
            # Calcular confiança
            confianca = max(proba)
            
            # Determinar categoria baseada na confiança
            if confianca >= 0.75:
                categoria = 'EXCELENTE'
            elif confianca >= 0.65:
                categoria = 'BOM'
            elif confianca >= 0.55:
                categoria = 'RAZOÁVEL'
            else:
                categoria = 'ARISCADO'
            
            resultado = {
                'prob_verdadeiro': proba[1],
                'prob_falso': proba[0],
                'predicao': 'VERDADEIRO' if predicao == 1 else 'FALSO',
                'confianca': confianca,
                'categoria': categoria,
                'features_usadas': list(X_pred.columns)
            }
            
            return resultado
            
        except Exception as e:
            self._log_detalhado(f"Erro ao analisar jogo: {e}", "ERRO")
            return None
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
        self._log_detalhado("Modelo salvo com sucesso", "SUCESSO")
    
    def _mostrar_ordenacao_aplicada(self, df_ordenado):
        """Mostrar exemplos da ordenação aplicada"""
        self._log_detalhado("📊 ORDENAÇÃO APLICADA NO ARQUIVO FINAL:")
        self._log_detalhado("   1️⃣  Data mais recente primeiro")
        self._log_detalhado("   2️⃣  Recomendação (EXCELENTE > BOA > REGULAR)") 
        self._log_detalhado("   3️⃣  Confiabilidade da Liga (ALTA > MEDIA > BAIXA)")
        self._log_detalhado("   4️⃣  Probabilidade de Sucesso (maior primeiro)")
        estatisticas = self._calcular_estatisticas_efetividade(df_ordenado)
        self._log_detalhado(f"📈 ESTATÍSTICAS DE EFETIVIDADE: {estatisticas}")
        # Mostrar primeiras linhas para verificar ordenação
        if len(df_ordenado) > 0:
            self._log_detalhado("🔍 PRIMEIRAS 5 LINHAS DO ARQUIVO ORDENADO:")
            for i in range(min(5, len(df_ordenado))):
                row = df_ordenado.iloc[i]
                self._log_detalhado(f"   {i+1}. {row['Date']} | {row['Recomendacao']} | {row['Liga_Categoria']} | {row['Probabilidade_Sucesso']:.1%} | {row['League']}")
    def _calcular_confiabilidade_por_desempenho(self, league_str):
        """
        Classificar liga baseada em desempenho histórico dos times
        """
        try:
            if not hasattr(self, 'df_treino') or self.df_treino is None:
                return 'MEDIA_CONFIABILIDADE'
            
            # Filtrar jogos da liga específica
            jogos_liga = self.df_treino[self.df_treino['League'] == league_str].copy()
            
            if len(jogos_liga) < 5:  # Poucos dados
                return 'MEDIA_CONFIABILIDADE'
            
            # ✅ CORREÇÃO: Extrair times do Stat se não existir coluna Time
            if 'Time' not in jogos_liga.columns:
                # Extrair times do Stat para análise de desempenho
                jogos_liga['Time_Extraido'] = jogos_liga['Stat'].apply(
                    lambda x: self._extrair_time_do_stat(x) if isinstance(x, str) else None
                )
                time_col = 'Time_Extraido'
            else:
                time_col = 'Time'
            
            # Verificar se temos times extraídos
            if time_col not in jogos_liga.columns or jogos_liga[time_col].isna().all():
                return 'MEDIA_CONFIABILIDADE'
            
            # Remover linhas sem time
            jogos_liga = jogos_liga.dropna(subset=[time_col])
            
            if len(jogos_liga) < 5:
                return 'MEDIA_CONFIABILIDADE'
            
            # ✅ CORREÇÃO: Verificar se temos coluna Target
            if 'Target' not in jogos_liga.columns:
                # Tentar criar target se não existir
                if 'Situação' in jogos_liga.columns:
                    situacao_map = {'VERDADEIRO': 1, 'Verdadeiro': 1, 'FALSO': 0, 'Falso': 0}
                    jogos_liga['Target'] = jogos_liga['Situação'].map(situacao_map)
                else:
                    return 'MEDIA_CONFIABILIDADE'
            
            # Calcular métricas por time
            try:
                streaks_liga = jogos_liga.groupby(time_col).agg({
                    'Target': ['mean', 'count'],
                    'Odds': 'mean'
                }).round(3)
                
                # Aplanar colunas multi-index
                streaks_liga.columns = ['Acuracia', 'Num_Jogos', 'Odds_Media']
                
                # Calcular confiabilidade geral da liga
                acuracia_media = streaks_liga['Acuracia'].mean()
                num_jogos_total = streaks_liga['Num_Jogos'].sum()
                
                # Classificar baseado na acurácia média
                if num_jogos_total >= 20:  # Dados suficientes
                    if acuracia_media >= 0.65:
                        return 'ALTA_CONFIABILIDADE'
                    elif acuracia_media >= 0.55:
                        return 'MEDIA_CONFIABILIDADE'
                    elif acuracia_media >= 0.45:
                        return 'BAIXA_CONFIABILIDADE'
                    else:
                        return 'MUITO_BAIXA_CONFIABILIDADE'
                else:
                    return 'MEDIA_CONFIABILIDADE'  # Dados insuficientes
                    
            except Exception as e:
                self._log_detalhado(f"Erro ao calcular confiabilidade: {e}", "ALERTA")
                return 'MEDIA_CONFIABILIDADE'
                
        except Exception as e:
            self._log_detalhado(f"Erro em calcular_confiabilidade: {e}", "ALERTA")
            return 'MEDIA_CONFIABILIDADE'
    def _extrair_time_do_stat(self, stat_text):
        """
        Extrair nome do time de um texto de estatística
        """
        if not isinstance(stat_text, str):
            return None
        
        # Padrões comuns para extrair times
        patterns = [
            r'^([A-Za-z\s\-\'\.]+) (?:have|has|had) ',  # "Team have..."
            r'^([A-Za-z\s\-\'\.]+) (?:won|lost|drew) ', # "Team won..."
            r'^([A-Za-z\s\-\'\.]+) in ',                # "Team in..."
        ]
        
        for pattern in patterns:
            match = re.search(pattern, stat_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    def _classificar_liga_fallback(self, league_str):
        """Fallback apenas para primeira execução sem dados"""
        # Lógica simples baseada em termos gerais
        if 'women' in league_str or 'nwsl' in league_str:
            return 'ALTA_CONFIABILIDADE'
        elif any(termo in league_str for termo in ['friendly', 'cup', 'test']):
            return 'BAIXA_CONFIABILIDADE'
        else:
            return 'MEDIA_CONFIABILIDADE'
    def _calcular_estatisticas_ligas(self):
        """Calcular estatísticas reais das ligas para debug"""
        if not hasattr(self, 'df_treino_limpo'):
            return
        
        stats_ligas = self.df_treino_limpo.groupby('League').agg({
            'Target': ['count', 'mean', 'std'],
            'Tamanho_Streak': ['mean', 'max'],
            'Odds': 'mean'
        }).round(3)
        
        self._log_detalhado("📊 ESTATÍSTICAS REAIS DAS LIGAS:")
        for liga in stats_ligas.index[:10]:  # Top 10 ligas
            dados = stats_ligas.loc[liga]
            total_jogos = dados[('Target', 'count')]
            taxa_acerto = dados[('Target', 'mean')]
            streak_medio = dados[('Tamanho_Streak', 'mean')]
            
            classificacao = self._classificar_liga(liga)
            self._log_detalhado(f"   {liga}: {total_jogos} jogos, {taxa_acerto:.1%} acertos, streak {streak_medio:.1f} → {classificacao}")
    def _converter_para_datetime_ordenacao(self, date_str):
        """Converter string de data para datetime para ordenação - VERSÃO ROBUSTA"""
        if pd.isna(date_str) or date_str in ['', 'DATA_INDISPONIVEL', 'NaT', None]:
            return pd.Timestamp.min  # Data mínima para ficar no final
        
        try:
            # Se já for datetime
            if isinstance(date_str, (pd.Timestamp, datetime)):
                return date_str
            
            date_str_clean = str(date_str).strip()
            
            # Formato brasileiro dd/mm/yyyy HH:MM
            if re.match(r'\d{2}/\d{2}/\d{4} \d{1,2}:\d{2}', date_str_clean):
                return datetime.strptime(date_str_clean, '%d/%m/%Y %H:%M')
            
            # Tentar conversão automática
            parsed_date = pd.to_datetime(date_str_clean, errors='coerce')
            if pd.isna(parsed_date):
                return pd.Timestamp.min
            return parsed_date
            
        except Exception:
            return pd.Timestamp.min
    def _converter_para_datetime(self, date_str):
        """Converter string de data para datetime para ordenação correta"""
        if pd.isna(date_str) or date_str in ['', 'DATA_INDISPONIVEL']:
            return pd.NaT
        
        try:
            # Tenta converter do formato brasileiro dd/mm/yyyy HH:MM
            if isinstance(date_str, str) and '/' in date_str:
                return datetime.strptime(date_str, '%d/%m/%Y %H:%M')
            # Tenta outros formatos
            return pd.to_datetime(date_str, errors='coerce')
        except:
            return pd.NaT
    def _analisar_jogo_avancado(self, row):
        """Análise avançada com modelo ML - VERSÃO CORRIGIDA COMPLETA"""
        try:
            # 1. Extrair features básicas
            tipo_estatistica = self._classificar_estatistica(row['Stat'])
            tamanho_streak = self._extrair_streak(row['Stat'])
            local_jogo = self._extrair_local(row['Next Match'])
            liga_categoria = self._classificar_liga(row['League'])
            
            # 2. Obter odds
            try:
                odds = float(row['Odds']) if pd.notna(row['Odds']) else 2.0
            except:
                odds = 2.0
            
            # 3. Verificar se temos modelo
            if not hasattr(self, 'model') or self.model is None:
                self._log_detalhado("Modelo não carregado", "ERRO")
                return self._analise_basica_futuro_melhorada(row)
            
            # 4. CRIAR FEATURES DICT COMPLETO
            features_dict = {}
            
            # ✅ CORREÇÃO CRÍTICA: GARANTIR QUE TODAS AS FEATURES EXISTEM
            # Features numéricas básicas
            features_dict['Odds'] = odds
            features_dict['Tamanho_Streak'] = float(tamanho_streak)
            features_dict['Odds_Streak_Interaction'] = odds * tamanho_streak
            features_dict['Prob_Implicita'] = 1 / odds if odds > 0 else 0.5
            
            # ✅ CORREÇÃO: ADICIONAR NOVAS FEATURES DE STREAK
            features_dict['Streak_Squared'] = float(tamanho_streak ** 2)
            
            # Para Streak_Tipo_Interaction, precisamos do encoded do tipo
            tipo_encoded = 0
            if hasattr(self, 'encoders') and 'stat_mapping' in self.encoders:
                tipo_encoded = float(self.encoders['stat_mapping'].get(tipo_estatistica, 0))
            features_dict['Streak_Tipo_Interaction'] = float(tamanho_streak * tipo_encoded)
            
            features_dict['Streak_Longo'] = 1.0 if tamanho_streak > 10 else 0.0
            features_dict['Streak_Curto'] = 1.0 if tamanho_streak <= 3 else 0.0
            features_dict['Streak_Medio'] = 1.0 if (tamanho_streak >= 4 and tamanho_streak <= 8) else 0.0
            
            # Features encoded
            # Tipo_Estatistica_encoded
            if hasattr(self, 'encoders') and 'stat_mapping' in self.encoders:
                features_dict['Tipo_Estatistica_encoded'] = float(
                    self.encoders['stat_mapping'].get(tipo_estatistica, -1)
                )
            else:
                features_dict['Tipo_Estatistica_encoded'] = -1.0
            
            # Local_Jogo_encoded
            if hasattr(self, 'encoders') and 'local_encoder' in self.encoders:
                encoder = self.encoders['local_encoder']
                if hasattr(encoder, 'classes_') and local_jogo in encoder.classes_:
                    features_dict['Local_Jogo_encoded'] = float(encoder.transform([local_jogo])[0])
                else:
                    features_dict['Local_Jogo_encoded'] = -1.0
            else:
                features_dict['Local_Jogo_encoded'] = -1.0
            
            # Liga_Categoria_encoded
            if hasattr(self, 'encoders') and 'liga_encoder' in self.encoders:
                encoder = self.encoders['liga_encoder']
                if hasattr(encoder, 'classes_') and liga_categoria in encoder.classes_:
                    features_dict['Liga_Categoria_encoded'] = float(encoder.transform([liga_categoria])[0])
                else:
                    features_dict['Liga_Categoria_encoded'] = -1.0
            else:
                features_dict['Liga_Categoria_encoded'] = -1.0
            
            # Streak_Categoria_encoded
            # Primeiro categorizar o streak
            def categorizar_streak(streak):
                if streak <= 3:
                    return 'Curto'
                elif streak <= 6:
                    return 'Medio'
                elif streak <= 10:
                    return 'Longo'
                else:
                    return 'Muito_Longo'
            
            streak_categoria = categorizar_streak(tamanho_streak)
            if hasattr(self, 'encoders') and 'streak_order' in self.encoders:
                features_dict['Streak_Categoria_encoded'] = float(
                    self.encoders['streak_order'].get(streak_categoria, -1)
                )
            else:
                features_dict['Streak_Categoria_encoded'] = -1.0
            
            # Dia_Semana
            dia_semana = -1
            if 'Date' in row and pd.notna(row['Date']) and row['Date'] != '':
                try:
                    if isinstance(row['Date'], str):
                        # Formato "13/12/2025 09:00"
                        date_obj = datetime.strptime(str(row['Date']), '%d/%m/%Y %H:%M')
                    else:
                        date_obj = pd.to_datetime(row['Date'])
                    dia_semana = date_obj.weekday()
                except:
                    dia_semana = -1
            features_dict['Dia_Semana'] = float(dia_semana)
            
            # Tipo_Performance
            if hasattr(self, 'encoders') and 'tipo_performance' in self.encoders:
                features_dict['Tipo_Performance'] = float(
                    self.encoders['tipo_performance'].get(tipo_estatistica, 0.5)
                )
            else:
                features_dict['Tipo_Performance'] = 0.5
            
            # ✅ DEBUG: Verificar features criadas
            self._log_detalhado(f"🔍 DEBUG - Features criadas para {row.get('Stat', '')[:30]}...:")
            self._log_detalhado(f"   Número de features: {len(features_dict)}")
            self._log_detalhado(f"   Keys: {list(features_dict.keys())[:5]}...")
            
            # 5. VERIFICAR SE TEMOS features_para_treino DEFINIDAS
            if not hasattr(self, 'features_para_treino') or not self.features_para_treino:
                self._log_detalhado("⚠️  Features para treino não definidas", "ALERTA")
                # Criar lista de features baseada no features_dict
                self.features_para_treino = list(features_dict.keys())
            
            # 6. CRIAR ARRAY NA ORDEM CORRETA
            X_array = []
            features_faltantes = []
            
            for feature in self.features_para_treino:
                if feature in features_dict:
                    X_array.append(features_dict[feature])
                else:
                    # Valor padrão para feature faltante
                    self._log_detalhado(f"⚠️  Feature '{feature}' não encontrada no dict", "ALERTA")
                    features_faltantes.append(feature)
                    # Valor padrão baseado no tipo
                    if feature.endswith('_encoded') or feature == 'Dia_Semana':
                        X_array.append(-1.0)
                    elif 'Streak' in feature:
                        X_array.append(0.0)
                    elif 'Odds' in feature or 'Prob' in feature:
                        X_array.append(1.0)
                    else:
                        X_array.append(0.0)
            
            if features_faltantes:
                self._log_detalhado(f"⚠️  Features faltantes: {features_faltantes}", "ALERTA")
            
            # ✅ VERIFICAÇÃO FINAL: Garantir que o array não está vazio
            if len(X_array) == 0:
                self._log_detalhado("❌ ERRO CRÍTICO: Array de features vazio!", "ERRO")
                return self._analise_basica_futuro_melhorada(row)
            
            X_pred = np.array([X_array])
            
            self._log_detalhado(f"✅ Array criado com shape: {X_pred.shape}", "SUCESSO")
            
            # 7. FAZER PREDIÇÃO
            try:
                probabilidade = self.model.predict_proba(X_pred)[0][1]  # Probabilidade de VERDADEIRO
                previsao = self.model.predict(X_pred)[0]
                
                self._log_detalhado(f"✅ Predição ML: prob={probabilidade:.3f}, pred={previsao}", "SUCESSO")
                
            except Exception as e:
                self._log_detalhado(f"❌ Erro no predict_proba: {e}", "ERRO")
                raise
            
            # 8. BÔNUS
            bonus_complementar = self._calcular_bonus_complementar(row['League'], row['Next Match'], row['Date'])
            bonus_confiabilidade = self._calcular_bonus_confiabilidade_liga(row['League'])
            
            # ✅ Bônus baseado no tamanho da streak
            bonus_streak = min(tamanho_streak * 0.015, 0.20)  # +1.5% por jogo, máximo 20%
            
            # Bônus baseado no tipo de estatística
            if tipo_estatistica == 'VITORIA':
                bonus_tipo = 0.08
            elif tipo_estatistica == 'BTTS':
                bonus_tipo = 0.05
            elif tipo_estatistica == 'OVER_2.5':
                bonus_tipo = 0.04
            else:
                bonus_tipo = 0.0
            
            bonus_total = bonus_complementar + bonus_confiabilidade + bonus_streak + bonus_tipo
            
            # 9. AJUSTAR PROBABILIDADE COM BÔNUS
            probabilidade_ajustada = min(probabilidade + bonus_total, 0.95)
            
            # 10. DECISÃO FINAL - CORREÇÃO IMPORTANTE
            # Se probabilidade > 0.6, SEMPRE predizer VERDADEIRO (manter streak)
            if probabilidade_ajustada > 0.60:
                previsao_final = 1  # VERDADEIRO
                previsao_str = 'VERDADEIRO'
            elif probabilidade_ajustada > 0.40:
                previsao_final = previsao  # Usar predição do modelo
                previsao_str = 'VERDADEIRO' if previsao == 1 else 'FALSO'
            else:
                previsao_final = 0  # FALSO
                previsao_str = 'FALSO'
            
            # Log para debug
            self._log_detalhado(f"📊 DECISÃO FINAL: prob_ajust={probabilidade_ajustada:.3f} → Previsão: {previsao_str}", "INFO")
            
            # 11. CLASSIFICAÇÃO
            padrao = self._classificar_padrao(probabilidade_ajustada)
            recomendacao = self._classificar_recomendacao(probabilidade_ajustada)
            
            # 12. ANÁLISE DETALHADA
            analise = f"{padrao}: {tipo_estatistica} em {tamanho_streak} jogos (Odds: {odds:.2f})"
            if bonus_streak > 0:
                analise += f" Streak bonus: +{bonus_streak*100:.0f}%"
            if bonus_tipo > 0:
                analise += f" Tipo bonus: +{bonus_tipo*100:.0f}%"
            analise += f" Previsão: {'MANTERÁ STREAK' if previsao_str == 'VERDADEIRO' else 'INTERROMPERÁ STREAK'}"
            
            return {
                'Probabilidade_Sucesso': probabilidade_ajustada,
                'Previsao': previsao_str,
                'Padrao': padrao,
                'Recomendacao': recomendacao,
                'Analise_Detalhada': analise,
                'Bonus_Total': bonus_total
            }
            
        except Exception as e:
            self._log_detalhado(f"❌ Erro em análise avançada: {str(e)}", "ERRO")
            import traceback
            self._log_detalhado(f"Detalhes: {traceback.format_exc()}", "ERRO")
            # Fallback robusto
            return self._analise_basica_futuro_melhorada(row)
    def _analise_basica_futuro_melhorada(self, row):
        """Análise básica melhorada - VERSÃO ROBUSTA"""
        try:
            # Extrair informações
            tipo_estatistica = self._classificar_estatistica(row.get('Stat', ''))
            tamanho_streak = self._extrair_streak(row.get('Stat', ''))
            
            # Calcular odds
            try:
                odds = float(row.get('Odds', 2.0)) if pd.notna(row.get('Odds')) else 2.0
            except:
                odds = 2.0
            
            # Probabilidade base
            prob_base = 0.5
            
            # Fatores
            if tamanho_streak >= 10:
                prob_base += 0.15
            elif tamanho_streak >= 7:
                prob_base += 0.10
            elif tamanho_streak >= 5:
                prob_base += 0.05
            
            if tipo_estatistica == 'VITORIA':
                prob_base += 0.08
            elif tipo_estatistica == 'BTTS':
                prob_base += 0.05
            elif tipo_estatistica == 'OVER_2.5':
                prob_base += 0.03
            
            if odds < 1.5:
                prob_base += 0.07
            elif odds < 2.0:
                prob_base += 0.03
            
            # Limitar
            prob_final = max(0.4, min(0.85, prob_base))
            
            # Decisão
            if prob_final >= 0.6:
                previsao = 'VERDADEIRO'
                padrao = 'PADRAO_SOLIDO' if prob_final >= 0.65 else 'PADRAO_REGULAR'
                recomendacao = 'EXCELENTE' if prob_final >= 0.75 else 'BOA'
            else:
                previsao = 'FALSO'
                padrao = 'PADRAO_REGULAR'
                recomendacao = 'REGULAR'
            
            return {
                'Probabilidade_Sucesso': round(prob_final, 3),
                'Previsao': previsao,
                'Padrao': padrao,
                'Recomendacao': recomendacao,
                'Analise_Detalhada': f'Análise básica: Streak de {tamanho_streak} jogos para {tipo_estatistica}',
                'Bonus_Total': 0.0
            }
        except Exception as e:
            self._log_detalhado(f"⚠️  Erro em análise básica: {e}", "ALERTA")
            # Fallback absoluto
            return {
                'Probabilidade_Sucesso': 0.6,
                'Previsao': 'VERDADEIRO',
                'Padrao': 'PADRAO_REGULAR',
                'Recomendacao': 'REGULAR',
                'Analise_Detalhada': 'Análise automática',
                'Bonus_Total': 0.0
            }
    def _gerar_multiplas_recomendadas(self, df_futuros, num_multiplas=7):
        """Gerar múltiplas de 2, 3 ou 4 times APENAS com confiança > 75% - VERSÃO OTIMIZADA"""
        self._log_detalhado("GERANDO MÚLTIPLAS DE ALTA CONFIANÇA (>75%):")
        self._log_detalhado("="*50)
            
        if df_futuros is None or len(df_futuros) == 0:
            self._log_detalhado("Dados futuros não disponíveis para gerar múltiplas", "ERRO")
            return []
        
        # FILTRAR APENAS OS MELHORES JOGOS PARA EVITAR COMBINAÇÕES EXCESSIVAS
        jogos_excelentes = df_futuros[df_futuros['Recomendacao'] == 'EXCELENTE']
        jogos_bons = df_futuros[df_futuros['Recomendacao'] == 'BOA']
        
        # LIMITAR O NÚMERO DE JOGOS CONSIDERADOS
        max_jogos_considerar = 15  # Reduzir drasticamente para evitar combinações excessivas
        jogos_excelentes = jogos_excelentes.head(max_jogos_considerar)
        jogos_bons = jogos_bons.head(max_jogos_considerar)
        
        # Combinar e ordenar por confiança
        todos_jogos = pd.concat([jogos_excelentes, jogos_bons])
        todos_jogos = todos_jogos.sort_values('Probabilidade_Sucesso', ascending=False)
        
        self._log_detalhado(f"Jogos excelentes: {len(jogos_excelentes)}, Jogos bons: {len(jogos_bons)}")
        
        if len(todos_jogos) < 2:
            self._log_detalhado("Número insuficiente de jogos para gerar múltiplas", "ERRO")
            return []
        
        # Criar lista de jogos com ID da partida - VERSÃO OTIMIZADA
        jogos_lista = []
        
        for idx, row in todos_jogos.iterrows():
            time = str(row.get('Time', 'TIME_DESCONHECIDO')).strip()
            mercado = str(row.get('Tipo_Estatistica', 'MERCADO_DESCONHECIDO')).strip()
            
            # Pular times desconhecidos
            if time == 'TIME_DESCONHECIDO':
                continue
                
            time = re.split(r' have', time, flags=re.IGNORECASE)[0]
            
            # Extrair ID único da partida
            id_partida = self._extrair_id_partida(row)
            
            jogos_lista.append({
                'id': f"{time}_{mercado}",
                'time': time,
                'time_adversario': str(row.get('Next Match', 'ADVERSARIO_DESCONHECIDO')).strip(),
                'data': str(row.get('Date','DATA DESCONHECIDA')).strip(),
                'mercado': mercado,
                'odds': float(row.get('Odds', 1.0)) if pd.notna(row.get('Odds')) else 1.0,
                'confianca': float(row.get('Probabilidade_Sucesso', 0)),
                'analise': str(row.get('Analise_Detalhada', '')),
                'id_partida': id_partida
            })
        
        # Remover duplicatas
        jogos_unicos = []
        ids_vistos = set()
        for jogo in jogos_lista:
            if jogo['id'] not in ids_vistos:
                ids_vistos.add(jogo['id'])
                jogos_unicos.append(jogo)
        
        # LIMITAR AINDA MAIS PARA EVITAR COMBINAÇÕES
        jogos_unicos = jogos_unicos[:12]  # Máximo 12 jogos para combinar
        
        self._log_detalhado(f"Jogos únicos para múltiplas: {len(jogos_unicos)}")
        
        if len(jogos_unicos) < 2:
            self._log_detalhado("Número insuficiente de jogos únicos para gerar múltiplas", "ERRO")
            return []
        
        # GERAR MÚLTIPLAS COM LIMITAÇÃO
        todas_multiplas = []
        
        # 1. Múltiplas de 2 jogos (mais eficiente)
        self._log_detalhado("Gerando múltiplas de 2 jogos...")
        multiplas_2 = self._gerar_multiplas_tamanho_fixo(jogos_unicos, 2, 20)  # Limitar a 20 combinações
        todas_multiplas.extend(multiplas_2)
        
        # 2. Múltiplas de 3 jogos (se houver jogos suficientes)
        if len(jogos_unicos) >= 3:
            self._log_detalhado("Gerando múltiplas de 3 jogos...")
            multiplas_3 = self._gerar_multiplas_tamanho_fixo(jogos_unicos, 3, 15)  # Limitar a 15 combinações
            todas_multiplas.extend(multiplas_3)
        
        # 3. Múltiplas de 4 jogos (apenas se muitos jogos bons)
        if len(jogos_unicos) >= 4 and len(multiplas_2) > 5:
            self._log_detalhado("Gerando múltiplas de 4 jogos...")
            multiplas_4 = self._gerar_multiplas_tamanho_fixo(jogos_unicos, 4, 10)  # Limitar a 10 combinações
            todas_multiplas.extend(multiplas_4)
        
        # FILTRAR POR CONFIANÇA MÍNIMA
        multiplas_validas = [m for m in todas_multiplas if m['confianca_media'] > 0.75]
        
        self._log_detalhado(f"Múltiplas válidas (>75%): {len(multiplas_validas)}")
        
        if len(multiplas_validas) == 0:
            self._log_detalhado("Nenhuma múltipla atingiu o mínimo de 75% de confiança", "ALERTA")
            return []
        
        # ORDENAR E LIMITAR RESULTADOS
        multiplas_validas.sort(key=lambda x: (x['confianca_media'], x['score']), reverse=True)
        multiplas_finais = multiplas_validas[:10]  # Máximo 10 múltiplas no resultado
        
        # CLASSIFICAR E EXIBIR
        multiplas_altissima = [m for m in multiplas_finais if m['confianca_media'] > 0.85]
        multiplas_alta = [m for m in multiplas_finais if m['confianca_media'] > 0.75]
        
        self._log_detalhado("CLASSIFICAÇÃO DAS MÚLTIPLAS (>75%):")
        self._log_detalhado(f"   ⭐⭐⭐ ALTÍSSIMA (>85%): {len(multiplas_altissima)} múltiplas")
        self._log_detalhado(f"   ⭐⭐ ALTA (>75%): {len(multiplas_alta)} múltiplas")
        
        # EXIBIR APENAS AS MELHORES
        self._exibir_multiplas_por_categoria(multiplas_altissima, "ALTÍSSIMA CONFIANÇA", "⭐⭐⭐", min(5, len(multiplas_altissima)))
        self._exibir_multiplas_por_categoria(multiplas_alta, "ALTA CONFIANÇA", "⭐⭐", min(5, len(multiplas_alta)))
        
        return multiplas_finais

    def _gerar_multiplas_tamanho_fixo(self, jogos, tamanho, max_combinacoes):
        """Gerar múltiplas de tamanho fixo com limite máximo - VERSÃO OTIMIZADA"""
        from itertools import combinations, islice
        
        multiplas = []
        combinacoes_geradas = 0
        
        # Usar islice para limitar o número de combinações geradas
        for combo in islice(combinations(jogos, tamanho), max_combinacoes):
            combinacoes_geradas += 1
            
            # Verificar conflitos
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
            
            # Calcular métricas da múltipla
            multipla = self._calcular_metricas_multipla(combo)
            if multipla['confianca_media'] > 0.75:  # Filtro mínimo
                multiplas.append(multipla)
        
        self._log_detalhado(f"   Múltiplas de {tamanho}: {len(multiplas)} válidas de {combinacoes_geradas} combinações testadas")
        return multiplas

    def _calcular_metricas_multipla(self, combinacao_jogos):
        """Calcular métricas para uma múltipla de qualquer tamanho - VERSÃO OTIMIZADA"""
        jogos_ordenados = sorted(combinacao_jogos, key=lambda x: x['id'])
        
        # Calcular odd total e confianças
        odd_total = 1.0
        confiancas = []
        
        for jogo in combinacao_jogos:
            odd = jogo['odds'] if pd.notna(jogo['odds']) and jogo['odds'] > 0 else 1.0
            odd_total *= odd
            confiancas.append(jogo['confianca'])
        
        # Calcular confiança média
        confianca_media = sum(confiancas) / len(confiancas)
        
        # Score simplificado para melhor performance
        score_final = confianca_media * 100
        
        return {
            'jogos': jogos_ordenados,
            'odd_total': odd_total,
            'confianca_media': confianca_media,
            'score': score_final,
            'tamanho': len(combinacao_jogos)
        }
    def _tem_conflito_logico(self, jogo1, jogo2):
            """Verificar se há conflito lógico entre dois jogos - VERSÃO RIGOROSA"""
            # Se são da mesma partida, NÃO PERMITIR NENHUMA COMBINAÇÃO
            if jogo1['id_partida'] == jogo2['id_partida'] and jogo1['id_partida'] != 'DESCONHECIDO':
                self._log_detalhado(f"CONFLITO: {jogo1['time']} e {jogo2['time']} são do mesmo jogo {jogo1['id_partida']}", "ALERTA")
                return True
            
            return False

    def _extrair_id_partida(self, row):
            """Extrair ID único da partida - VERSÃO DEFINITIVA CORRIGIDA"""
            import re
            stat = str(row.get('Stat', ''))
            next_match = str(row.get('Next Match', ''))
            local = re.search(r'\b(home|away)\b', next_match, re.IGNORECASE)
            
            # EXTRAIR TIME PRINCIPAL DA COLUNA STAT
            time_principal = 'TIME_DESCONHECIDO'
            
            # Padrão: "Nome do Time have/has/had ..."
            if ' have ' in stat:
                time_principal = stat.split(' have ')[0].strip()
            elif ' has ' in stat:
                time_principal = stat.split(' has ')[0].strip()
            elif ' had ' in stat:
                time_principal = stat.split(' had ')[0].strip()
            
            # DEBUG: Mostrar o que foi extraído
            self._log_detalhado(f"Extraindo: Stat='{stat[:50]}...' → Time='{time_principal}'")
            
            # EXTRAIR INFORMAÇÕES DO NEXT MATCH
            if ' vs ' in next_match.lower():
                partes = next_match.lower().split(' vs ')
                
                if len(partes) == 2:
                    local = next_match.lower().split(' vs ')[0].strip() # "home" ou "away"
                    adversario = partes[1].strip()
                    
                    # Formatar adversário (primeira letra maiúscula em cada palavra)
                    adversario = ' '.join(word.capitalize() for word in adversario.split())
                    
                    self._log_detalhado(f"Next Match: '{next_match}' → Local='{local}', Adversário='{adversario}'")
                    
                    # Determinar times baseado no local
                    if time_principal != 'TIME_DESCONHECIDO':
                        if local == 'home':
                            time_casa = time_principal
                            time_visitante = adversario
                        elif local == 'away':
                            time_casa = adversario
                            time_visitante = time_principal
                        else:
                            # Se não é home/away claro, assumir que time_principal é o primeiro
                            time_casa = time_principal
                            time_visitante = adversario
                        
                        # Criar ID único ordenando os times alfabeticamente
                        times_ordenados = sorted([time_casa, time_visitante])
                        id_partida = f"{times_ordenados[0]}_vs_{times_ordenados[1]}"
                        
                        self._log_detalhado(f"ID Partida criado: {id_partida}")
                        return id_partida
            
            self._log_detalhado("Não foi possível extrair ID da partida", "ALERTA")
            return 'DESCONHECIDO'

        
    def _exibir_multiplas_por_categoria(self, multiplas, categoria, icone, max_exibir):
            """Exibir múltiplas por categoria de confiança - VERSÃO CORRIGIDA"""
            
            if not multiplas:
                return
            
            self._log_detalhado(f"{icone} {categoria} {icone}")
            self._log_detalhado("=" * 50)
            
            for i, multipla in enumerate(multiplas[:max_exibir], 1):
                tamanho_str = {
                    2: "DUPLA",
                    3: "TRIPLA", 
                    4: "QUÁDRUPLA"
                }.get(multipla['tamanho'], f"{multipla['tamanho']} JOGOS")
                
                self._log_detalhado(f"🔮 {tamanho_str} {i}:")
                self._log_detalhado(f"   Odd Total: {multipla['odd_total']:.2f}")
                self._log_detalhado(f"   Confiança Média: {multipla['confianca_media']:.1%} ⭐")
                self._log_detalhado(f"   Nº de Seleções: {multipla['tamanho']}")
                
                for jogo in multipla['jogos']:
                    # CORREÇÃO: Usar 'time' em vez de 'id' para mostrar o nome real
                    time_nome = jogo.get('time', 'TIME_DESCONHECIDO')
                    mercado = jogo.get('mercado', 'MERCADO_DESCONHECIDO')
                    odd_str = f"@{jogo['odds']:.2f}" if pd.notna(jogo['odds']) and jogo['odds'] > 0 else "@nan"
                    conf_individual = f"({jogo['confianca']:.1%})"
                    self._log_detalhado(f"   ✅ {time_nome} - {mercado} {odd_str} {conf_individual}")
                self._log_detalhado("-" * 40)
    def criar_backup_manual(self):
            """Criar backup manual do modelo atual"""
            if not os.path.exists(self.modelo_path):
                self._log_detalhado("Nenhum modelo encontrado para backup", "ALERTA")
                return False
            
            self._fazer_backup_modelo()
            self._log_detalhado("Backup manual criado com sucesso", "SUCESSO")
            return True
    # EXECUÇÃO PRINCIPAL INTELIGENTE
    def verificar_arquivos_config(self,config):
            """Verificar se arquivos de configuração existem"""
            arquivos_ok = True
            
            for nome, caminho in config.items():
                if 'base' in nome and not os.path.exists(caminho):
                    print(f"❌ Arquivo não encontrado: {caminho}")
                    arquivos_ok = False
                else:
                    print(f"✅ {nome}: {caminho}")
            
            return arquivos_ok
    def gerar_previsoes_futuras(self, output_path='previsoes_evolutivas.csv'):
        """Gerar previsões para jogos futuros com recomendações - VERSÃO COM STAT"""
        # ✅ CORREÇÃO 1: Verificar e carregar modelo robustamente
        if not hasattr(self, 'model') or self.model is None:
            self._log_detalhado("Modelo não detectado. Tentando carregar...", "INFO")
            if not self.carregar_modelo():
                self._log_detalhado("⚠️  Modelo não carregado. Tentando treinar novo...", "ALERTA")
                if not self.treinar_modelo_evolutivo():
                    self._log_detalhado("❌ Falha crítica: não foi possível carregar nem treinar modelo", "ERRO")
                    return None
        
        if self.model is None:
            self._log_detalhado("❌ Modelo não carregado após todas as tentativas", "ERRO")
            return None
        
        if self.base_futuros_path is None:
            self._log_detalhado("❌ Caminho da base de futuros não especificado", "ERRO")
            return None
        
        self._log_detalhado("GERANDO PREVISÕES INTELIGENTES...")
        
        # ✅ CORREÇÃO 2: Carregar dados futuros com validação robusta
        df_futuros = self.carregar_dados(self.base_futuros_path)
        if df_futuros is None or len(df_futuros) == 0:
            self._log_detalhado("❌ Falha ao carregar dados futuros", "ERRO")
            return None
        
        self._log_detalhado(f"✅ Dados futuros carregados: {len(df_futuros)} registros")
        
        # ✅ IMPORTANTE: SALVAR UMA CÓPIA DA COLUNA STAT ORIGINAL ANTES DE PROCESSAR
        if 'Stat' in df_futuros.columns:
            df_futuros['Stat_Original'] = df_futuros['Stat'].copy()
            self._log_detalhado("✅ Coluna Stat original preservada como 'Stat_Original'")
        
        # ✅ CORREÇÃO 3: Preparar dados futuros com tratamento de erro
        try:
            df_futuros = self._preparar_dados_futuros(df_futuros)
            if df_futuros is None or len(df_futuros) == 0:
                self._log_detalhado("❌ Nenhum dado futuro válido após preparação", "ERRO")
                return None
            self._log_detalhado(f"✅ Dados futuros preparados: {len(df_futuros)} registros válidos")
        except Exception as e:
            self._log_detalhado(f"❌ Erro ao preparar dados futuros: {e}", "ERRO")
            return None
        
        # ✅ CORREÇÃO 4: Gerar previsões com fallback robusto
        previsoes_detalhadas = []
        erros_analise = 0
        
        for idx, row in df_futuros.iterrows():
            try:
                # Primeiro tentar análise avançada
                try:
                    previsao = self._analisar_jogo_avancado(row)
                    previsoes_detalhadas.append(previsao)
                except Exception as e_interno:
                    self._log_detalhado(f"⚠️  Análise avançada falhou jogo {idx}. Tentando fallback...", "ALERTA")
                    # Fallback para análise básica MAS com probabilidade variável
                    previsao = self._analise_basica_futuro_melhorada(row)
                    previsoes_detalhadas.append(previsao)
                    erros_analise += 1
                    
            except Exception as e:
                self._log_detalhado(f"❌ Erro crítico ao analisar jogo {idx}: {e}", "ERRO")
                # Fallback com probabilidade baseada em odds
                prob_variada = self._calcular_probabilidade_odds(row.get('Odds', 2.0))
                previsoes_detalhadas.append({
                    'Probabilidade_Sucesso': prob_variada,
                    'Previsao': 'VERDADEIRO',
                    'Padrao': 'PADRAO_REGULAR',
                    'Recomendacao': 'REGULAR',
                    'Analise_Detalhada': 'Análise automática - odds consideradas',
                    'Bonus_Total': 0.0
                })
                erros_analise += 1
        
        if erros_analise > 0:
            self._log_detalhado(f"⚠️  {erros_analise} jogos tiveram erros na análise (usando fallback)", "ALERTA")
        
        # ✅ CORREÇÃO 5: Adicionar previsões ao DataFrame com validação
        colunas_previsao = ['Probabilidade_Sucesso', 'Previsao', 'Padrao', 'Recomendacao', 'Analise_Detalhada', 'Bonus_Total']
        
        for col in colunas_previsao:
            valores = []
            for previsao in previsoes_detalhadas:
                if col in previsao:
                    valores.append(previsao[col])
                else:
                    # Valor padrão se a coluna não existir
                    if col == 'Probabilidade_Sucesso':
                        valores.append(0.5)
                    elif col == 'Previsao':
                        valores.append('INDETERMINADO')
                    elif col == 'Padrao':
                        valores.append('PADRAO_REGULAR')
                    elif col == 'Recomendacao':
                        valores.append('REGULAR')
                    elif col == 'Analise_Detalhada':
                        valores.append('Análise não disponível')
                    else:
                        valores.append(0.0)
            df_futuros[col] = valores
        
        # ✅ CORREÇÃO 6: Adicionar coluna de efetividade
        self._log_detalhado("Aplicando ordenação personalizada...")
        try:
            df_futuros = self._adicionar_coluna_efetividade(df_futuros)
        except Exception as e:
            self._log_detalhado(f"⚠️  Erro ao adicionar efetividade: {e}", "ALERTA")
            df_futuros['Efetividade'] = 'JOGO_FUTURO'
        
        # ✅ CORREÇÃO 7: Ordenação final com tratamento robusto
        try:
            df_futuros_ordenado = self._aplicar_ordenacao_final(df_futuros)
        except Exception as e:
            self._log_detalhado(f"⚠️  Erro na ordenação: {e}. Usando ordenação padrão...", "ALERTA")
            # Ordenação de fallback
            if 'Date' in df_futuros.columns:
                df_futuros['Data_Ordenacao'] = df_futuros['Date'].apply(self._converter_para_datetime_ordenacao)
                df_futuros = df_futuros.sort_values('Data_Ordenacao', ascending=True)
                df_futuros = df_futuros.drop('Data_Ordenacao', axis=1)
            df_futuros_ordenado = df_futuros
        
        # ✅ CORREÇÃO 8: DEFINIR COLUNAS FINAIS COM STAT
        # Lista de colunas base (incluindo Stat agora)
        colunas_base = [
            'League', 
            'Stat',  # ✅ AGORA INCLUÍDO AQUI
            'Next Match', 
            'Odds', 
            'Date', 
            'Resultado', 
            'Situação',
            'Tipo_Estatistica', 
            'Liga_Categoria'
        ]
        
        # ✅ SE A COLUNA Stat NÃO EXISTIR, USAR A ORIGINAL
        if 'Stat' not in df_futuros_ordenado.columns and 'Stat_Original' in df_futuros_ordenado.columns:
            df_futuros_ordenado['Stat'] = df_futuros_ordenado['Stat_Original']
            self._log_detalhado("✅ Coluna Stat restaurada da original", "SUCESSO")
        
        colunas_analise = [
            'Probabilidade_Sucesso', 
            'Efetividade', 
            'Previsao', 
            'Padrao', 
            'Recomendacao', 
            'Analise_Detalhada'
        ]
        
        # Garantir que todas as colunas existam
        colunas_finais = []
        for col in colunas_base + colunas_analise:
            if col in df_futuros_ordenado.columns:
                colunas_finais.append(col)
            else:
                self._log_detalhado(f"⚠️  Coluna '{col}' não encontrada no DataFrame", "ALERTA")
        
        # Criar DataFrame final apenas com colunas existentes
        df_final = df_futuros_ordenado[colunas_finais].copy()
        
        # ✅ CORREÇÃO 9: GARANTIR QUE STAT ESTÁ NA POSIÇÃO CORRETA
        # Se Stat existe, mover para após League (posição 1)
        if 'Stat' in df_final.columns:
            cols = list(df_final.columns)
            # Encontrar posição de League
            if 'League' in cols:
                league_idx = cols.index('League')
                # Mover Stat para logo após League
                cols.insert(league_idx + 1, cols.pop(cols.index('Stat')))
                df_final = df_final[cols]
                self._log_detalhado("✅ Coluna Stat posicionada após League", "SUCESSO")
        
        # ✅ CORREÇÃO 10: Ordenação no DataFrame final
        try:
            df_final = self._aplicar_ordenacao_final(df_final)
        except:
            pass  # Se falhar, manter como está
        
        # ✅ CORREÇÃO 11: Gerar múltiplas recomendadas
        try:
            self._log_detalhado("\n🔮 GERANDO MÚLTIPLAS RECOMENDADAS...")
            df_para_multiplas = df_futuros.copy()
            self._gerar_multiplas_recomendadas(df_para_multiplas)
        except Exception as e:
            self._log_detalhado(f"⚠️  Erro ao gerar múltiplas: {e}", "ALERTA")
        
        # ✅ CORREÇÃO 12: Salvar arquivo com encoding adequado
        try:
            df_final.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')
            self._log_detalhado(f"✅ Previsões salvas em: {output_path}", "SUCESSO")
            self._log_detalhado(f"📊 Total de jogos analisados: {len(df_final)}")
            
            # Mostrar estrutura do arquivo salvo
            self._log_detalhado("\n📋 ESTRUTURA DO ARQUIVO SALVO:")
            for i, col in enumerate(df_final.columns, 1):
                self._log_detalhado(f"   {i:2d}. {col}")
            
            # Estatísticas detalhadas
            if 'Recomendacao' in df_final.columns:
                excelentes = len(df_final[df_final['Recomendacao'] == 'EXCELENTE'])
                bons = len(df_final[df_final['Recomendacao'] == 'BOA'])
                regulares = len(df_final[df_final['Recomendacao'] == 'REGULAR'])
                
                self._log_detalhado(f"\n⭐ Jogos EXCELENTES: {excelentes} ({excelentes/len(df_final)*100:.1f}%)")
                self._log_detalhado(f"👍 Jogos BONS: {bons} ({bons/len(df_final)*100:.1f}%)")
                self._log_detalhado(f"📊 Jogos REGULARES: {regulares} ({regulares/len(df_final)*100:.1f}%)")
                
                # Distribuição de padrões
                if 'Padrao' in df_final.columns:
                    padroes = df_final['Padrao'].value_counts()
                    self._log_detalhado("\n📈 DISTRIBUIÇÃO DE PADRÕES:")
                    for padrao, count in padroes.items():
                        self._log_detalhado(f"   {padrao}: {count} jogos")
            
            # ✅ CORREÇÃO 13: Mostrar ordenação aplicada
            try:
                self._mostrar_ordenacao_aplicada(df_final)
            except Exception as e:
                self._log_detalhado(f"⚠️  Não foi possível mostrar ordenação: {e}", "ALERTA")
            
            # ✅ CORREÇÃO 14: Criar arquivo resumo das melhores oportunidades COM STAT
            self._criar_arquivo_resumo(df_final, output_path.replace('.csv', '_resumo.csv'))
            
            return df_final
            
        except Exception as e:
            self._log_detalhado(f"❌ Erro ao salvar arquivo: {e}", "ERRO")
            # Tentar salvar em formato alternativo
            try:
                df_final.to_csv(output_path, index=False, sep=',', encoding='utf-8')
                self._log_detalhado(f"✅ Arquivo salvo com separador alternativo", "SUCESSO")
                return df_final
            except:
                self._log_detalhado("❌ Falha completa ao salvar arquivo", "ERRO")
                return None
    def _criar_arquivo_resumo(self, df, output_path):
        """Versão simplificada para evitar erro"""
        try:
            if len(df) == 0:
                return
            
            # Apenas salvar um resumo básico com as melhores oportunidades
            jogos_ordenados = df.sort_values(['Probabilidade_Sucesso', 'Recomendacao'], 
                                        ascending=[False, True])
            
            # Pegar top 20
            top_20 = jogos_ordenados.head(20).copy()
            
            # Salvar
            top_20.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')
            self._log_detalhado(f"✅ Resumo básico salvo: {output_path}", "SUCESSO")
            
        except Exception as e:
            self._log_detalhado(f"⚠️  Não foi possível criar resumo: {e}", "ALERTA")
    def _calcular_probabilidade_odds(self, odds):
        """Calcular probabilidade baseada nas odds"""
        try:
            odds_float = float(odds) if pd.notna(odds) else 2.0
            # Fórmula básica: probabilidade = 1/odds * fator_correcao
            prob_base = 1 / odds_float if odds_float > 0 else 0.5
            # Ajustar para faixa realista (0.4 a 0.8)
            prob_ajustada = max(0.4, min(0.8, prob_base * 1.3))
            return round(prob_ajustada, 3)
        except:
            return 0.6  # Fallback
    # ✅ CORREÇÃO 14: Adicionar método para criar arquivo resumo
    
    def _aplicar_ordenacao_final(self, df):
        """Aplicar ordenação final ao DataFrame - PRESERVANDO STAT"""
        # Criar colunas temporárias para ordenação
        df_ordenacao = df.copy()
        
        # ✅ VERIFICAR SE STAT EXISTE
        if 'Stat' not in df_ordenacao.columns and 'Stat_Original' in df_ordenacao.columns:
            df_ordenacao['Stat'] = df_ordenacao['Stat_Original']
        
        # Garantir que todas as colunas importantes existem
        if 'Resultado' not in df_ordenacao:
            df_ordenacao['Resultado'] = ''
        if 'Situação' not in df_ordenacao:
            df_ordenacao['Situação'] = ''
        if 'Tipo_Estatistica' in df_ordenacao.columns:
            # Mover Tipo_Estatistica para posição correta
            df_ordenacao.insert(8, 'Tipo_Estatistica', df_ordenacao.pop('Tipo_Estatistica'))
        
        # 1. Converter datas para datetime
        df_ordenacao['Data_Ordenacao'] = df_ordenacao['Date'].apply(self._converter_para_datetime_ordenacao)
        
        # 2. Criar ordem customizada para Recomendacao
        ordem_recomendacao = {'EXCELENTE': 1, 'BOA': 2, 'REGULAR': 3}
        df_ordenacao['Ordem_Recomendacao'] = df_ordenacao['Recomendacao'].map(ordem_recomendacao).fillna(4)
        
        # 3. Criar ordem customizada para Liga_Categoria
        ordem_liga = {'ALTA_CONFIABILIDADE': 1, 'MEDIA_CONFIABILIDADE': 2, 'BAIXA_CONFIABILIDADE': 3}
        df_ordenacao['Ordem_Liga'] = df_ordenacao['Liga_Categoria'].map(ordem_liga).fillna(4)
        
        # 4. Ordenação FINAL conforme solicitado
        df_ordenado = df_ordenacao.sort_values([
            'Data_Ordenacao',           # 1º: Data mais recente primeiro
            'Ordem_Recomendacao',       # 2º: Melhor recomendação
            'Ordem_Liga',               # 3º: Alta confiabilidade primeiro
            'Probabilidade_Sucesso'     # 4º: Maior probabilidade
        ], ascending=[
            True,  # Data: mais antiga primeiro
            True,   # Recomendacao: Excelente(1) antes de Boa(2)
            True,   # Liga: Alta(1) antes de Media(2)
            False   # Probabilidade: maior primeiro
        ])
        
        # Remover colunas temporárias de ordenação
        colunas_para_manter = []
        for col in df_ordenado.columns:
            if not col.startswith('Ordem_') and col != 'Data_Ordenacao':
                colunas_para_manter.append(col)
        
        return df_ordenado[colunas_para_manter]
    
if __name__ == "__main__":
    print("🤖 SISTEMA DE ANÁLISE EVOLUTIVA DE APOSTAS - VERSÃO CORRIGIDA")
    print("="*50)
    
    # CONFIGURAÇÃO
    config = {
        'base_treino': r'..\scraping_futebol\base_dados_total.csv',
        'base_futuros': r'..\scraping_futebol\scraping\adam choi_dados_20260413_173947.csv',
        'modelo_salvo': r'modelo_apostas_evolutivo copy.joblib'
    }
    
    try:
        analisador = AnalisadorApostasEvolutivo(
            base_treino_path=config['base_treino'],
            base_futuros_path=config['base_futuros'],
            modelo_path=config['modelo_salvo']
        )
        
        if analisador.verificar_arquivos_config(config):
            print("\n💾 Criando backup manual...")
            analisador.criar_backup_manual()
            
            print("\n1. 🔄 Carregando/treinando modelo...")
            # ✅ CORREÇÃO: Primeiro tenta carregar o modelo existente
            if not analisador.carregar_modelo():
                print("   ⚠️  Modelo não encontrado ou inválido. Treinando novo...")
                if not analisador.treinar_modelo_evolutivo():
                    print("❌ Falha crítica: não foi possível carregar nem treinar modelo")
                    exit(1)
            
            print("\n2. 🔮 Gerando previsões...")
            previsoes = analisador.gerar_previsoes_futuras('previsoes_evolutivas.csv')
            
            print("\n" + "="*50)
            print("✅ PROCESSO CONCLUÍDO!")
        else:
            print("❌ Verifique os caminhos dos arquivos")
            
    except Exception as e:
        print(f"💥 Erro crítico: {e}")
        import traceback
        traceback.print_exc()