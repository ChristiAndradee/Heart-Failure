import os
import numpy as np
import pandas as pd
from pickle import load

from dados import CONTINUAS, BINARIAS, PASTA_MODELOS


class Inferidor:
    """Sistema inteligente de agrupamento: recebe os dados de um paciente
    desconhecido, aplica o MESMO pre-processamento do treino (normaliza continuas
    com o normalizador salvo; binarias em 0/1) e indica a qual GRUPO ele pertence,
    com um GRAU DE SIMILARIDADE baseado nas distancias aos centroides."""

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
        distancias = self.modelo.transform(X)          # distancia a cada centroide
        # grau de similaridade: proximidade (1/(1+d)) normalizada entre os grupos
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
    # AVISO: Ana Laura (9a), Ryann (21a), Kelvi (19a), Haboski (23a) e Enache (22a)
    # estao abaixo da idade minima da base de treino (40 anos). Idades ajustadas para
    # 40 anos com pequenas variacoes para demonstracao; resultados nao tem validade
    # clinica para pacientes reais nessa faixa etaria.
    pacientes_novos = pd.DataFrame([
        {  # Ana Laura, 9a -> ajustada para 40a | F, anemia, fracao de ejecao limítrofe
            'age': 40, 'creatinine_phosphokinase': 180, 'ejection_fraction': 32,
            'platelets': 220000, 'serum_creatinine': 0.8, 'serum_sodium': 136,
            'anaemia': 1, 'diabetes': 0, 'high_blood_pressure': 0, 'sex': 0, 'smoking': 0,
        },
        {  # Ryann, 21a -> ajustado para 42a | M, diabetico, exames proximos do normal
            'age': 42, 'creatinine_phosphokinase': 310, 'ejection_fraction': 38,
            'platelets': 262000, 'serum_creatinine': 1.1, 'serum_sodium': 137,
            'anaemia': 0, 'diabetes': 1, 'high_blood_pressure': 0, 'sex': 1, 'smoking': 0,
        },
        {  # Kelvi, 19a -> ajustado para 40a | M, fumante, pressao alta
            'age': 40, 'creatinine_phosphokinase': 420, 'ejection_fraction': 35,
            'platelets': 245000, 'serum_creatinine': 1.2, 'serum_sodium': 135,
            'anaemia': 0, 'diabetes': 0, 'high_blood_pressure': 1, 'sex': 1, 'smoking': 1,
        },
        {  # Haboski, 23a -> ajustado para 43a | M, anemia e diabetes, sodio baixo
            'age': 43, 'creatinine_phosphokinase': 95, 'ejection_fraction': 30,
            'platelets': 198000, 'serum_creatinine': 1.4, 'serum_sodium': 129,
            'anaemia': 1, 'diabetes': 1, 'high_blood_pressure': 0, 'sex': 1, 'smoking': 0,
        },
        {  # Enache, 22a -> ajustado para 42a | M, fumante, CPK elevado
            'age': 42, 'creatinine_phosphokinase': 860, 'ejection_fraction': 40,
            'platelets': 303000, 'serum_creatinine': 0.9, 'serum_sodium': 140,
            'anaemia': 0, 'diabetes': 0, 'high_blood_pressure': 0, 'sex': 1, 'smoking': 1,
        },
        {  # Cida, 51a | F, pressao alta, diabetica, fracao de ejecao reduzida
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
