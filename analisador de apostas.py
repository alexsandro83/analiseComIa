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
        if 'Situacao' in df.columns and 'Previsao' in df.columns and 'Date' in df.columns:
            df['Efetividade'] = df.apply(
                lambda row: self._calcular_efetividade(
                    row['Situacao'], 
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
            self._log_detalhado("Colunas Situacao, Previsao ou Date não encontradas", "ALERTA")
        
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
        colunas_essenciais = ['Situacao', 'Odds', 'Stat']
        for col in colunas_essenciais:
            if col not in self.df_treino.columns:
                raise ValueError(f"Coluna essencial '{col}' não encontrada")
        
        # Verificar valores nulos
        nulos = self.df_treino[colunas_essenciais].isnull().sum()
        if nulos.any():
            self._log_detalhado(f"Valores nulos encontrados: {nulos}", "ALERTA")
    
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
    def corrigir_caracteres_especiais_csv(self,camiho_arquivo,df):
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

        # Aplicar correções em todas as colunas de texto
        for coluna in df.columns:
            if df[coluna].dtype == 'object':
                for errado, correto in correcoes.items():
                    df[coluna] = df[coluna].str.replace(errado, correto, regex=False)

        # Salvar o DataFrame corrigido
        df.to_csv(f'{camiho_arquivo}', index=False, encoding='utf-8')
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df = df.drop_duplicates(subset=['League', 'Stat', 'Next Match', 'Date', 'Resultado', 'Situacao'], keep='first')
        df = df.sort_values(['Date', 'Stat'], ascending=[True, False], inplace=True)
        print("Caracteres especiais corrigidos e arquivo salvo!")
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
    def treinar_modelo_evolutivo(self, forcar_retreino=False):
        """Treinar ou atualizar modelo com dados mais recentes"""
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
        
        # Treinar novo modelo
        self._log_detalhado("Treinando novo modelo com dados atualizados...")
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
            
            # Avaliação detalhada
            self._avaliar_modelo_detalhado(X_test, y_test)
            
            # ✅ MODIFICAÇÃO: Fazer backup apenas se acurácia melhorar
            backup_feito = self._fazer_backup_modelo(self.acuracia_modelo)
            
            # Salvar modelo (sempre salva o novo, mas só faz backup se melhorou)
            self._salvar_modelo()
            
            self._log_detalhado(f"Modelo treinado com sucesso! Acurácia: {accuracy:.2%}", "SUCESSO")
            if backup_feito:
                self._log_detalhado("✅ Backup automático criado (acurácia melhorou)", "SUCESSO")
            
            self._log_detalhado(f"Total de amostras de treino: {len(self.df_treino_limpo)}")
            
            return True
            
        except Exception as e:
            self._log_detalhado(f"Erro no treinamento: {e}", "ERRO")
            return False
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
        if 'Situacao' not in df_atual.columns:
            self._log_detalhado("Coluna 'Situacao' não encontrada após carregamento", "ERRO")
            self._log_detalhado(f"Colunas disponíveis: {list(df_atual.columns)}")
            return True
        
        # Filtrar dados válidos
        df_atual_limpo = df_atual[df_atual['Situacao'].notna() & 
                                (df_atual['Situacao'] != '')]
        
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

    def _preparar_dados_treino(self):
        """Preparar dados para treino"""
        self._log_detalhado("Preparando dados para treino...")

        # DEBUG: Mostrar informações dos dados
        self._log_detalhado(f"Colunas disponíveis: {list(self.df_treino.columns)}")
        self._log_detalhado(f"Primeiras linhas:")
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
        
        # CORREÇÃO: Usar coluna 'Situacao' que existe na sua base
        if 'Situacao' in self.df_treino.columns:
            situacao_map = {'VERDADEIRO': 1, 'Verdadeiro': 1, 'FALSO': 0, 'Falso': 0, 'GREEN': 1, 'RED': 0, 'WIN': 1, 'LOSS': 0}
            self.df_treino['Target'] = self.df_treino['Situacao'].map(situacao_map).fillna(-1)
            self._log_detalhado(f"Target criado: {len(self.df_treino[self.df_treino['Target'] != -1])} amostras válidas")
        else:
            self._log_detalhado("Coluna 'Situacao' não encontrada - ESSENCIAL", "ERRO")
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

        # Filtrar apenas dados com target válido
        self.df_treino_limpo = self.df_treino[self.df_treino['Target'] != -1].copy()
        self._log_detalhado(f"Dados finais para treino: {len(self.df_treino_limpo)} registros válidos")
        
        # Mostrar distribuição
        if len(self.df_treino_limpo) > 0:
            verd_count = (self.df_treino_limpo['Target'] == 1).sum()
            fals_count = (self.df_treino_limpo['Target'] == 0).sum()
            self._log_detalhado(f"Distribuição: VERDADEIRO={verd_count}, FALSO={fals_count}")
        
            # ✅ ADICIONAR APÓS CRIAR A COLUNA TARGET:
            # Adicionar coluna de efetividade para dados de treino
        self.df_treino = self._adicionar_coluna_efetividade(self.df_treino)
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
        
        # Adicionar coluna Situacao vazia para consistência
        if 'Situacao' not in df_futuros.columns:
            df_futuros['Situacao'] = ''
            self._log_detalhado("Coluna Situacao adicionada (vazia para jogos futuros)")
        
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
            self._log_detalhado(f"Modelo carregado - Acurácia: {self.acuracia_modelo:.2%}", "SUCESSO")
            return True
        except Exception as e:
            self._log_detalhado(f"Erro ao carregar modelo: {e}", "ERRO")
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
        """Gerar múltiplas de 2, 3 ou 4 times APENAS com confiança > 75% - VERSÃO CORRIGIDA"""
        self._log_detalhado("GERANDO MÚLTIPLAS DE ALTA CONFIANÇA (>75%):")
        self._log_detalhado("="*50)
            
        if df_futuros is None or len(df_futuros) == 0:
            self._log_detalhado("Dados futuros não disponíveis para gerar múltiplas", "ERRO")
            return []
        
        jogos_excelentes = df_futuros[df_futuros['Recomendacao'] == 'EXCELENTE']
        jogos_bons = df_futuros[df_futuros['Recomendacao'] == 'BOA']
        
        # Combinar e ordenar por confiança
        todos_jogos = pd.concat([jogos_excelentes, jogos_bons])
        todos_jogos = todos_jogos.sort_values('Probabilidade_Sucesso', ascending=False)
        
        self._log_detalhado(f"Jogos excelentes: {len(jogos_excelentes)}, Jogos bons: {len(jogos_bons)}")
        
        if len(todos_jogos) < 2:
            self._log_detalhado("Número insuficiente de jogos para gerar múltiplas", "ERRO")
            return []
        
        # DEBUG: Verificar se a coluna Time existe e tem valores
        self._log_detalhado("DEBUG - Verificando coluna 'Time' no DataFrame:")
        if 'Time' in todos_jogos.columns:
            self._log_detalhado(f"   Coluna 'Time' encontrada")
            self._log_detalhado(f"   Valores únicos: {todos_jogos['Time'].nunique()} times")
            self._log_detalhado(f"   Primeiros 5 valores: {todos_jogos['Time'].head().tolist()}")
        else:
            self._log_detalhado("   COLUNA 'TIME' NÃO ENCONTRADA - PROBLEMA CRÍTICO", "ERRO")
            return []
        
        # Criar lista de jogos com ID da partida - VERSÃO CORRIGIDA
        jogos_lista = []
        
        for idx, row in todos_jogos.iterrows():
            # CORREÇÃO: Garantir que estamos pegando o time CORRETAMENTE
            time = str(row.get('Time', 'TIME_DESCONHECIDO')).strip()
            mercado = str(row.get('Tipo_Estatistica', 'MERCADO_DESCONHECIDO')).strip()
            
            # DEBUG: Verificar cada jogo sendo processado
            if time == 'TIME_DESCONHECIDO':
                self._log_detalhado(f"ALERTA: Time desconhecido encontrado - Stat: '{row.get('Stat', '')[:50]}...'", "ALERTA")
            
            # Extrair ID único da partida
            id_partida = self._extrair_id_partida(row)
            
            jogos_lista.append({
                'id': f"{time}_{mercado}",  # ID único para evitar duplicatas
                'time': time,  # Nome real do time - GARANTIDO
                'time_adversario':str(row.get('Next Match', 'ADVERSARIO_DESCONHECIDO')).strip(),
                'data':str(row.get('Date','DATA DESCONHECIDA')).strip(),
                'mercado': mercado,
                'odds': float(row.get('Odds', 1.0)) if pd.notna(row.get('Odds')) else 1.0,
                'confianca': float(row.get('Probabilidade_Sucesso', 0)),
                'analise': str(row.get('Analise_Detalhada', '')),
                'id_partida': id_partida
            })
                # Extrai tudo antes de "have"
            
        # DEBUG: Mostrar alguns jogos extraídos (FORA DO LOOP)
        self._log_detalhado(f"Primeiros 10 jogos extraídos:")
        for i, jogo in enumerate(jogos_lista[:10]):
            jogo['time'] = re.split(r' have', jogo['time'], flags=re.IGNORECASE)[0]
            self._log_detalhado(f"   {i+1}. {jogo['data']} {jogo['time']} - {jogo['time_adversario']} {jogo['mercado']} {jogo['odds']} (Conf: {jogo['confianca']:.1%})")
        
        # Remover duplicatas
        jogos_unicos = []
        ids_vistos = set()
        for jogo in jogos_lista:
            if jogo['id'] not in ids_vistos:
                ids_vistos.add(jogo['id'])
                jogos_unicos.append(jogo)
        
        self._log_detalhado(f"Jogos únicos disponíveis: {len(jogos_unicos)}")
        
        if len(jogos_unicos) < 2:
            self._log_detalhado("Número insuficiente de jogos únicos para gerar múltiplas", "ERRO")
            return []
        
        # 🔥 NOVA LÓGICA: IDENTIFICAR JOGOS COM CONFLITO MAS CRIAR MÚLTIPLAS SEPARADAS
        jogos_para_multiplas = []
        grupos_conflito = {}
        
        # Identificar grupos de conflito (times do mesmo jogo)
        for jogo in jogos_unicos:
            if jogo['id_partida'] != 'DESCONHECIDO':
                if jogo['id_partida'] not in grupos_conflito:
                    grupos_conflito[jogo['id_partida']] = []
                grupos_conflito[jogo['id_partida']].append(jogo)
        
        self._log_detalhado("Grupos de conflito identificados:")
        for partida_id, jogos in grupos_conflito.items():
            if len(jogos) > 1:
                times = [f"{j['time']}({j['confianca']:.1%})" for j in jogos]
                self._log_detalhado(f"   {partida_id} {jogo['mercado']}: {', '.join(times)}")
        
        # MANTER TODOS OS JOGOS, mas marcar os que têm conflito
        for jogo in jogos_unicos:
            jogo['tem_conflito'] = False
            if jogo['id_partida'] in grupos_conflito and len(grupos_conflito[jogo['id_partida']]) > 1:
                jogo['tem_conflito'] = True
            jogos_para_multiplas.append(jogo)
        
        self._log_detalhado(f"Jogos para múltiplas: {len(jogos_para_multiplas)} (incluindo opções de conflito)")
        
        if len(jogos_para_multiplas) < 2:
            self._log_detalhado("Número insuficiente de jogos para gerar múltiplas", "ERRO")
            return []
        
        # GERAR MÚLTIPLAS EVITANDO CONFLITOS DIRETOS
        todas_multiplas = []
        from itertools import combinations
        
        # 1. Múltiplas de 2 jogos
        self._log_detalhado("Gerando múltiplas de 2 jogos (evitando conflitos diretos)...")
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
        
        self._log_detalhado(f"   Múltiplas de 2 válidas: {multiplas_validas_2}")
        
        # 2. Múltiplas de 3 jogos
        multiplas_validas_3 = 0
        if len(jogos_para_multiplas) >= 3:
            self._log_detalhado("Gerando múltiplas de 3 jogos (evitando conflitos diretos)...")
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
            
            self._log_detalhado(f"   Múltiplas de 3 válidas: {multiplas_validas_3}")
        
        # 3. Múltiplas de 4 jogos
        multiplas_validas_4 = 0
        if len(jogos_para_multiplas) >= 4:
            self._log_detalhado("Gerando múltiplas de 4 jogos (evitando conflitos diretos)...")
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
            
            self._log_detalhado(f"   Múltiplas de 4 válidas: {multiplas_validas_4}")
        
        self._log_detalhado(f"Múltiplas geradas com confiança > 75%: {len(todas_multiplas)}")
        
        if len(todas_multiplas) == 0:
            self._log_detalhado("Nenhuma múltipla atingiu o mínimo de 75% de confiança", "ALERTA")
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
        
        self._log_detalhado("CLASSIFICAÇÃO DAS MÚLTIPLAS (>75%):")
        self._log_detalhado(f"   ⭐⭐⭐ ALTÍSSIMA (>85%): {len(multiplas_altissima)} múltiplas")
        self._log_detalhado(f"   ⭐⭐ ALTA (>75%): {len(multiplas_alta)} múltiplas")
        self._log_detalhado(f"   Distribuição: 2-jogos={multiplas_validas_2}, 3-jogos={multiplas_validas_3}, 4-jogos={multiplas_validas_4}")
        
        # EXIBIR POR CATEGORIA
        self._exibir_multiplas_por_categoria(multiplas_altissima, "ALTÍSSIMA CONFIANÇA", "⭐⭐⭐", min(10, len(multiplas_altissima)))
        self._exibir_multiplas_por_categoria(multiplas_alta, "ALTA CONFIANÇA", "⭐⭐", min(10, len(multiplas_alta)))
        
        return multiplas_unicas

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
    def _aplicar_ordenacao_final(self, df):
        """Aplicar ordenação final ao DataFrame"""
        # Criar colunas temporárias para ordenação
        df_ordenacao = df.copy()
        
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
            False,  # Data: mais recente primeiro
            True,   # Recomendacao: Excelente(1) antes de Boa(2)
            True,   # Liga: Alta(1) antes de Media(2)
            False   # Probabilidade: maior primeiro
        ])
        
        # Remover colunas temporárias de ordenação
        colunas_finais = [col for col in df_ordenado.columns if not col.startswith('Ordem_') and col != 'Data_Ordenacao']
        return df_ordenado[colunas_finais]
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
        """Gerar previsões para jogos futuros com recomendações"""
        if self.model is None:
            self._log_detalhado("Modelo não carregado. Execute treinamento primeiro.", "ERRO")
            return
        
        if self.base_futuros_path is None:
            self._log_detalhado("Caminho da base de futuros não especificado", "ERRO")
            return
            
        self._log_detalhado("GERANDO PREVISÕES INTELIGENTES...")
        
        # Carregar dados futuros
        df_futuros = self.carregar_dados(self.base_futuros_path)
        if df_futuros is None or len(df_futuros) == 0:
            self._log_detalhado("Falha ao carregar dados futuros", "ERRO")
            return
        
        df_futuros = self._preparar_dados_futuros(df_futuros)
        
        # Gerar previsões
        previsoes_detalhadas = []
        
        for idx, row in df_futuros.iterrows():
            try:
                previsao = self._analisar_jogo_avancado(row)
                previsoes_detalhadas.append(previsao)
            except Exception as e:
                self._log_detalhado(f"Erro ao analisar jogo {idx}: {e}", "ALERTA")
                previsoes_detalhadas.append(self._analise_basica_futuro(row))
        
        # Adicionar previsões ao DataFrame - APENAS AS COLUNAS EXISTENTES
        colunas_previsao = ['Probabilidade_Sucesso', 'Previsao', 'Padrao', 'Recomendacao', 'Analise_Detalhada', 'Bonus_Total']
        for col in colunas_previsao:
            df_futuros[col] = [p[col] for p in previsoes_detalhadas]
        
        # ✅ CORREÇÃO: ORDENAÇÃO FINAL ANTES DE SALVAR
        self._log_detalhado("Aplicando ordenação personalizada...")
        df_futuros = self._adicionar_coluna_efetividade(df_futuros)
        # 1. Converter datas para datetime para ordenação correta
        df_futuros = self._aplicar_ordenacao_final(df_futuros)
        
        # REMOVER COLUNAS TEMPORÁRIAS QUE NÃO DEVEM APARECER NO CSV FINAL
        colunas_para_manter = [
            'League', 'Stat', 'Next Match', 'Odds', 'Date', 'Situacao',
            'Tipo_Estatistica', 'Liga_Categoria',
            'Probabilidade_Sucesso','Efetividade', 'Previsao', 'Padrao', 'Recomendacao', 'Analise_Detalhada', 
        ]

        # Manter apenas as colunas que existem no DataFrame
        colunas_existentes = [col for col in colunas_para_manter if col in df_futuros.columns]
        df_futuros_ordenado = df_futuros[colunas_existentes].copy()
        
        # ✅ CORREÇÃO: GARANTIR que a ordenação está aplicada no DataFrame final
        df_futuros_ordenado = self._aplicar_ordenacao_final(df_futuros_ordenado)
        
        # Gerar múltiplas recomendadas (usando o DataFrame original para evitar perda de colunas)
        self._gerar_multiplas_recomendadas(df_futuros)
        
        # ✅ CORREÇÃO: SALVAR O DATAFRAME ORDENADO
        df_futuros_ordenado.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')
        self._log_detalhado(f"Previsões salvas em: {output_path}", "SUCESSO")
        self._log_detalhado(f"Total de jogos analisados: {len(df_futuros_ordenado)}")
        self._log_detalhado(f"Jogos EXCELENTES: {len(df_futuros_ordenado[df_futuros_ordenado['Recomendacao'] == 'EXCELENTE'])}")
        self._log_detalhado(f"Jogos BONS: {len(df_futuros_ordenado[df_futuros_ordenado['Recomendacao'] == 'BOA'])}")
        
        # Mostrar ordenação aplicada
        self._mostrar_ordenacao_aplicada(df_futuros_ordenado)
        
        return df_futuros_ordenado
if __name__ == "__main__":
    print("🤖 SISTEMA DE ANÁLISE EVOLUTIVA DE APOSTAS")
    print("="*50)
    
    # CONFIGURAÇÃO - AJUSTE ESTES CAMINHOS
    config = {
        'base_treino': r'D:\Downloads\scraping_futebol\base_dados_total.csv',
        'base_futuros': r'd:\Downloads\scraping_futebol\scraping\adam choi_dados_20251027_221137.csv',
        'modelo_salvo': 'modelo_apostas_evolutivo.joblib'
    }
    
    try:
        # 1. Primeiro criar o analisador
        analisador = AnalisadorApostasEvolutivo(
            base_treino_path=config['base_treino'],
            base_futuros_path=config['base_futuros'],
            modelo_path=config['modelo_salvo']
        )
        
        # 2. ✅ CORREÇÃO: Agora chamar o método DA INSTÂNCIA
        if analisador.verificar_arquivos_config(config):
            
            # 3. Backup manual
            print("\n💾 Criando backup manual do modelo atual...")
            analisador.criar_backup_manual()
            
            # 4. Treinar/Atualizar modelo
            print("\n1. 🎯 Treinando modelo com dados atualizados...")
            sucesso_treino = analisador.treinar_modelo_evolutivo()
            
            if sucesso_treino:
                # 5. Gerar previsões
                print("\n2. 🔮 Gerando previsões inteligentes...")
                previsoes = analisador.gerar_previsoes_futuras('previsoes_evolutivas.csv')
                
                print("\n" + "="*50)
                print("✅ PROCESSO CONCLUÍDO!")
                print("="*50)
                print("📁 Arquivos gerados:")
                print("   - modelo_apostas_evolutivo.joblib (modelo treinado)")
                print("   - previsoes_evolutivas.csv (previsões + múltiplas)")
            else:
                print("❌ Falha no treinamento do modelo")
        else:
            print("❌ Verifique os caminhos dos arquivos de configuração")
            
    except Exception as e:
        print(f"💥 Erro crítico: {e}")
        import traceback
        traceback.print_exc()