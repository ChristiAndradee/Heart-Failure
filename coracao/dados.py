import os
import pandas as pd
from pickle import dump
from sklearn.preprocessing import MinMaxScaler

PASTA = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(PASTA)
PASTA_DATASETS = os.path.join(RAIZ, 'datasets')
PASTA_MODELOS = os.path.join(RAIZ, 'models')
PASTA_PLOTS = os.path.join(RAIZ, 'plots')

ARQUIVO = 'heart_failure_clinical_records_dataset.csv'

CONTINUAS = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
             'platelets', 'serum_creatinine', 'serum_sodium']

BINARIAS = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

EXCLUIR = ['time', 'DEATH_EVENT']

# Mapeamento semantico das variaveis binarias (0/1 -> rotulo legivel).
# Usado apenas na exibicao dos resultados; nao afeta o pre-processamento nem o modelo.
MAPA_BINARIAS = {
    'anaemia':            {0: 'Nao', 1: 'Sim'},
    'diabetes':           {0: 'Nao', 1: 'Sim'},
    'high_blood_pressure': {0: 'Nao', 1: 'Sim'},
    'sex':                {0: 'F',   1: 'M'},
    'smoking':            {0: 'Nao', 1: 'Sim'},
}

class PreparadorDados:

    def __init__(self):
        self.caminho = os.path.join(PASTA_DATASETS, ARQUIVO)
        self.colunas = CONTINUAS + BINARIAS

    def carregar(self):
        df = pd.read_csv(self.caminho)
        print(f'Base carregada -> {df.shape[0]} pacientes, {df.shape[1]} colunas '
              f'| nulos: {int(df.isnull().sum().sum())}')
        return df

    def preparar(self, df):
        normalizador = MinMaxScaler().fit(df[CONTINUAS])
        persistir_modelo(normalizador, 'normalizador')
        continuas_norm = pd.DataFrame(
            normalizador.transform(df[CONTINUAS]), columns=CONTINUAS, index=df.index)

        binarias = df[BINARIAS].astype(float)

        X = continuas_norm.join(binarias)[self.colunas]
        persistir_modelo(self.colunas, 'colunas')

        print(f'Pre-processamento: {len(CONTINUAS)} continuas normalizadas [0,1] + '
              f'{len(BINARIAS)} binarias 0/1 = {X.shape[1]} atributos de agrupamento')
        print(f'Excluidos do agrupamento: {EXCLUIR} (desfecho/tempo de estudo)\n')
        return X


def persistir_modelo(objeto, nome):
    os.makedirs(PASTA_MODELOS, exist_ok=True)
    dump(objeto, open(os.path.join(PASTA_MODELOS, f'Coracao_{nome}.pkl'), 'wb'))
