import os
import pandas as pd
from pickle import load

from dados import (PreparadorDados, persistir_modelo, CONTINUAS, BINARIAS,
                   PASTA_MODELOS, MAPA_BINARIAS)

pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 30)

def _carregar(nome):
    return load(open(os.path.join(PASTA_MODELOS, f'Coracao_{nome}.pkl'), 'rb'))


def descrever(df, modelo):
    normalizador = _carregar('normalizador')
    colunas = _carregar('colunas')

    centroides = pd.DataFrame(modelo.cluster_centers_, columns=colunas)

    continuas_reais = pd.DataFrame(
        normalizador.inverse_transform(centroides[CONTINUAS]), columns=CONTINUAS)
    binarias_prop = centroides[BINARIAS].abs()

    perfil = continuas_reais.join(binarias_prop).round(2)

    prep = PreparadorDados()
    X = prep.preparar(df)
    rotulos = modelo.predict(X)
    perfil.insert(0, 'n_pacientes', pd.Series(rotulos).value_counts().sort_index().values)
    perfil['mortalidade_%'] = (pd.Series(df['DEATH_EVENT'].values)
                               .groupby(rotulos).mean().values * 100).round(1)
    perfil.index.name = 'grupo'

    perfil_exibicao = perfil.copy()
    for col in BINARIAS:
        perfil_exibicao[col] = perfil[col].apply(
            lambda p: f"{MAPA_BINARIAS[col][round(p)]} ({p:.0%})"
        )

    print('\n=== Perfil dos grupos (centroides em unidades reais) ===')
    print(perfil_exibicao.to_string())
    return perfil

if __name__ == "__main__":
    prep = PreparadorDados()
    df = prep.carregar()
    modelo = _carregar('modelo_kmeans')
    descrever(df, modelo)
