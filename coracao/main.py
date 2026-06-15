from dados import PreparadorDados
from treinamento import TreinadorCluster
from descrever_grupos import descrever

if __name__ == "__main__":
    print('==================================================')
    print('  Analise - insuficiencia cardiaca)')
    print('  Metaestimador: KMeans')
    print('==================================================\n')

    #preprocessamento
    prep = PreparadorDados()
    df = prep.carregar()
    X = prep.preparar(df)

    #Aplicação do cotovelo e treino do KMeans
    treinador = TreinadorCluster()
    k = treinador.numero_otimo_clusters(X)
    modelo = treinador.treinar(X, k)

    #descricao dos grupos
    descrever(df, modelo)
