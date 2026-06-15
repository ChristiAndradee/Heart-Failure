import os
import numpy as np
import pandas as pd
from pickle import load

from dados import CONTINUAS, BINARIAS, PASTA_MODELOS

class Inferidor:

    def __init__(self, pasta_modelos=None):
        self.pasta = pasta_modelos or PASTA_MODELOS
        self.modelo = self._carregar('modelo_kmeans')
        self.normalizador = self._carregar('normalizador')
        self.colunas = self._carregar('colunas')

    def _carregar(self, nome):
        return load(open(os.path.join(self.pasta, f'Coracao_{nome}.pkl'), 'rb'))

    def _preparar(self, df_pacientes):
        cont = pd.DataFrame(self.normalizador.transform(df_pacientes[CONTINUAS]),
                            columns=CONTINUAS, index=df_pacientes.index)
        binr = df_pacientes[BINARIAS].astype(float)
        return cont.join(binr)[self.colunas]

    def indicar(self, df_pacientes):
        X = self._preparar(df_pacientes)
        grupos = self.modelo.predict(X)
        distancias = self.modelo.transform(X)
        proximidade = 1.0 / (1.0 + distancias)
        pertinencia = proximidade / proximidade.sum(axis=1, keepdims=True)
        return grupos, pertinencia, distancias

    def relatar(self, df_pacientes):
        grupos, pertinencia, distancias = self.indicar(df_pacientes)
        for i in range(len(df_pacientes)):
            g = int(grupos[i])
            print(f'\n=== Paciente {i + 1} ===')
            print(f'  Grupo indicado: {g}  '
                  f'(similaridade {pertinencia[i][g]:.1%} | distancia ao centroide {distancias[i][g]:.3f})')
            ordem = np.argsort(distancias[i])
            segundo = ordem[1]
            print(f'  2o grupo mais proximo: {segundo} (similaridade {pertinencia[i][segundo]:.1%})')


if __name__ == "__main__":

    pacientes_novos = pd.DataFrame([
        {  #Ana Laura
            'age': 40, 'creatinine_phosphokinase': 180, 'ejection_fraction': 32,
            'platelets': 220000, 'serum_creatinine': 0.8, 'serum_sodium': 136,
            'anaemia': 1, 'diabetes': 0, 'high_blood_pressure': 0, 'sex': 0, 'smoking': 0,
        },
        {  #Ryann
            'age': 42, 'creatinine_phosphokinase': 310, 'ejection_fraction': 38,
            'platelets': 262000, 'serum_creatinine': 1.1, 'serum_sodium': 137,
            'anaemia': 0, 'diabetes': 1, 'high_blood_pressure': 0, 'sex': 1, 'smoking': 0,
        },
        {  #Kelvi
            'age': 40, 'creatinine_phosphokinase': 420, 'ejection_fraction': 35,
            'platelets': 245000, 'serum_creatinine': 1.2, 'serum_sodium': 135,
            'anaemia': 0, 'diabetes': 0, 'high_blood_pressure': 1, 'sex': 1, 'smoking': 1,
        },
        {  #Haboski
            'age': 43, 'creatinine_phosphokinase': 95, 'ejection_fraction': 30,
            'platelets': 198000, 'serum_creatinine': 1.4, 'serum_sodium': 129,
            'anaemia': 1, 'diabetes': 1, 'high_blood_pressure': 0, 'sex': 1, 'smoking': 0,
        },
        {  #Enache
            'age': 42, 'creatinine_phosphokinase': 860, 'ejection_fraction': 40,
            'platelets': 303000, 'serum_creatinine': 0.9, 'serum_sodium': 140,
            'anaemia': 0, 'diabetes': 0, 'high_blood_pressure': 0, 'sex': 1, 'smoking': 1,
        },
        {  #Cida
            'age': 51, 'creatinine_phosphokinase': 140, 'ejection_fraction': 25,
            'platelets': 271000, 'serum_creatinine': 1.6, 'serum_sodium': 133,
            'anaemia': 0, 'diabetes': 1, 'high_blood_pressure': 1, 'sex': 0, 'smoking': 0,
        },
    ])

    nomes = ['Ana Laura', 'Ryann', 'Kelvi', 'Haboski', 'Enache', 'Cida']

    inferidor = Inferidor()
    grupos, pertinencia, distancias = inferidor.indicar(pacientes_novos)

    print('=== SISTEMA INTELIGENTE — agrupamento de pacientes ===\n')
    for i, nome in enumerate(nomes):
        g = int(grupos[i])
        segundo = int(np.argsort(distancias[i])[1])
        print(f'--- {nome} ---')
        print(f'  Grupo indicado : {g}  '
              f'(similaridade {pertinencia[i][g]:.1%} | distancia ao centroide {distancias[i][g]:.3f})')
        print(f'  2o mais proximo: {segundo} (similaridade {pertinencia[i][segundo]:.1%})')
