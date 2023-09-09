# MÓDULO DE PROCESSAMENTO

import pyspark

from pyspark.sql.functions import (
    col,
    asc,
    count,
    desc,
    isnull,
    lit,
    to_date,
    countDistinct,
    when,
    add_months,
    udf,
    length,
    unix_timestamp,
    from_unixtime,
    date_format,
    year,
    lpad,
    trim,
    max as sparkMax,
    round as sparkRound,
    sum as sparkSum,
    min as sparkMin,
)

from pyspark.sql.types import DateType, DecimalType, IntegerType, StringType, FloatType

from pyspark.sql import DataFrame as SparkDataFrame

from datetime import datetime

from babel.numbers import format_currency

###############################################
# Colunas de Data
###############################################


def criar_col_dat_hoje(
    df: SparkDataFrame, nm_col: str
) -> SparkDataFrame:
    """Adicao de uma coluna contendo a data de hoje

    Parâmetros:

        df (SparkDataFrame): dataframe em que será criada a coluna

        nm_col (str): string do nome da coluna a ser criada

    Retorno:

        SparkDataFrame: dataframe contendo a coluna com a data de hoje
    """
    return df.withColumn(nm_col, to_date(lit(datetime.now().strftime("%Y-%m-%d"))))


def filtrar_ultimos_n_meses(
    df: SparkDataFrame, nm_col_filtrada: str, n_meses: int
) -> SparkDataFrame:
    """Filtro dos n ultimos meses de uma coluna de data passada como parâmetro

    Parâmetros:

        df (SparkDataFrame): dataframe a ser filtrado

        nm_col_filtrada (str): string contendo o nome da coluna de datas usada como referência para o filtro

        n_meses (int): quantidade de meses a compor o período filtrado

    Retorno:

        SparkDataFrame: dataframe filtrado pelos últimos n meses
    """
    data_inicial = obter_data_inicial(df, nm_col_filtrada, n_meses)
    return df.filter(col(nm_col_filtrada).cast(DateType()) > data_inicial)


def obter_data_inicial(
    df: SparkDataFrame, nm_col_filtrada: str, n_meses: int
) -> datetime.date:
    """Cálculo da data inicial de um período de n meses a partir da data mais recente de uma coluna de datas.

    Parâmetros:

        df (SparkDataFrame): dataframe que contém a coluna de datas da qual será obtida a data inicial

        nm_col_filtrada (str): string contendo o nome da coluna de datas de referência

        n_meses (int): quantidade de meses a compor o período do qual se obtém a data inicial

    Retorno:

        datetime: data inicial do período de n meses
    """
    return df.agg(add_months(sparkMax(nm_col_filtrada), -n_meses)).collect()[0][0]

def obter_distrib_data(
    df: SparkDataFrame, lst_cols_dat: list
) -> SparkDataFrame:
    """Cálculo da distribuição de uma ou mais colunas de data. Retorna um dataframe contendo medidas descritivas da(s) coluna(s) passada(s) como parâmetro: valor mínimo, valor máximo e quartis.

    Parâmetros:
        df (SparkDataFrame): dataframe que contém a(s) coluna(s) de data
        cols_dat (list): lista de strings contendo o(s) nome(s) da(s) coluna(s) de data da(s) qual(is) será obtida a distribuição

    Retorno:
        SparkDataFrame: dataframe contendo a distribuição da(s) coluna(s) de data
    """
    
    return(
        df
       .select([unix_timestamp(c).alias(c) for c in lst_cols_dat])
       .summary()
       .filter(col('summary').isin('min','25%','50%','75%','max'))
       .select('summary',*[from_unixtime(c).cast(DateType()).alias(c) for c in lst_cols_dat]))


###############################################
# Colunas numéricas
###############################################


def obter_distrib(
    df: SparkDataFrame, cols_num
) -> SparkDataFrame:
    """Cálculo da distribuição de uma ou mais colunas numéricas. Retorna um dataframe contendo medidas descritivas da(s) coluna(s) passada(s) como parâmetro: contagem de registros, valor mínimo, valor máximo, média, desvio padrão e quartis.

    Parâmetros:
        df (SparkDataFrame): dataframe que contém a coluna numérica da qual será obtida a distribuição
        cols_num (_type_): string ou lista de strings contendo o(s) nome(s) da(s) coluna(s) numérica(s) da(s) qual(is) será obtida a distribuição

    Retorno:
        SparkDataFrame: dataframe contendo a distribuição da(s) coluna(s) numérica(s)
    """
    return df.select(cols_num).summary()

def obter_intervalo(
    df: SparkDataFrame, nm_col_num: str
) -> SparkDataFrame:
    """Identificação do intervalo de valores de uma coluna numérica. Retorna um dataframe contendo o menor e o maior valor contidos em uma coluna numérica, cujo nome é passado como parâmetro.

    Parâmetros:
        df (SparkDataFrame): dataframe que contém a coluna numérica da qual será obtido o intervalo
        nm_col_num (str): string contendo o nome da coluna numérica da qual será obtido o intervalo

    Retorno:
        SparkDataFrame: dataframe contendo o intervalo da coluna numérica
    """
    
    return df.agg(sparkMin(col(nm_col_num)).alias('valor_inicial'),sparkMax(col(nm_col_num)).alias('valor_final'))
    

###############################################
# Util
###############################################


def renomear_cols(
    df: SparkDataFrame, dict_cols: dict
) -> SparkDataFrame:
    """Funcao responsavel por renomear colunas.

    Parâmetros:

        df (SparkDataFrame): dataframe que contém as colunas a serem renomeadas

        dict_cols (dict): dicionário em que as chaves são os nomes originais e os valores são os novos nomes

    Retorno:

        SparkDataFrame: dataframe que contém as colunas renomeadas
    """
    cols_final = df.columns
    for i, c in enumerate(df.columns):
        try:
            cols_final[i] = list(dict_cols.values())[list(dict_cols.keys()).index(c)]
        except:
            cols_final[i] = c
    return df.select(
        [
            col(c_antes).alias(c_depois)
            for c_antes, c_depois in zip(df.columns, cols_final)
        ]
    )

def remover_cols_hudi(df):
    return df.select(df.columns[5:])

def renomear_cols_para_minusculo(
    df: SparkDataFrame,
) -> SparkDataFrame:
    """Essa função retorna o dataframe original com as colunas renomeadas para sua forma minúscula

    Parâmetros:

        df (SparkDataFrame): dataframe que contém as colunas a serem renomeadas

    Retorno:

        SparkDataFrame: dataframe que contém as colunas renomeadas
    """
    return df.select(*[col(df_col).alias(df_col.lower()) for df_col in df.columns])


def obter_pct(
    df: SparkDataFrame, lst_nm_cols: list, total: int
) -> SparkDataFrame:
    """Cálculo da porcentagem em relacao ao total dos registros das colunas numericas de uma lista.

    Parâmetros:

        df (SparkDataFrame): dataframe que contém a porcentagem das colunas em relação ao tamanho do dataframe

        lst_nm_cols (list): lista de strings contendo os nomes das colunas cuja porcentagem será calculada. 
            As colunas precisam ser numéricas para que a função funcione.

        total (int): Tamanho total que corresponde a 100% do que se quer calcular. 
        'Por exemplo, a quantidade total de registros ou o somatório dos valores de uma coluna numérica.

    Retorno:

        SparkDataFrame: dataframe contendo os valores das colunas do dataframe original em forma de porcentagem
    """

    return df.select(
        [sparkRound((col(c) * 100) / total, 2).alias(c) for c in lst_nm_cols]
    )


def formatar_cols_moeda(
    df: SparkDataFrame, lst_cols_moeda: list
) -> SparkDataFrame:
    """Formatação de colunas passadas como parâmetro para o formato de moeda que, no caso, é o Real, ou seja R$X,XX. A formatação não altera nem a ordem nem o nome das colunas do dataframe original.

    Parâmetros:
        df (SparkDataFrame): dataframe a ter colunas formatadas para tipo de moeda, no caso, o Real
        lst_cols_moeda (list): lista de strings contendo os nomes das colunas a serem formatadas para Real. As colunas precisam ser numéricas para que a função funcione

    Retorno:
        SparkDataFrame: dataframe contendo as colunas, cujos nomes foram passados como parâmetro, formatadas para moeda (no caso, Real)
    """

    format_currency_udf = udf(lambda a: format_currency(a, "BRL"))

    lst_cols_rest = [c for c in df.columns if c not in lst_cols_moeda]
    return df.select(
        *lst_cols_rest, *[format_currency_udf(c).alias(c) for c in lst_cols_moeda]
    )


def formatar_cols_decimal(
    df: SparkDataFrame, lst_cols_decimal: list, n_digitos: int = 2
) -> SparkDataFrame:
    """Formatação de colunas passadas como parâmetro para o formato de decimal, em que a quantidade de casas depois da vírgula é passada como parâmetro. A formatação não altera nem a ordem nem o nome das colunas do dataframe original.

    Parâmetros:
        df (SparkDataFrame): dataframe a ter colunas formatadas para decimal, com número de casas depois da vírgula passado como parâmetro
        lst_cols_decimal (list): lista de strings contendo os nomes das colunas a serem formatadas para decimal. As colunas precisam ser numéricas para que a função funcione
        n_digitos (int, optional): número inteiro que indica a quantidade de dígitos depois da vírgula do número decimal. Por padrão, é 2.

    Retorno:
        SparkDataFrame: dataframe contendo as colunas, cujos nomes foram passados como parâmetro, formatadas para decimal com número de casas depois da vírgula passado como parâmetro
    """

    lst_cols_rest = [c for c in df.columns if c not in lst_cols_decimal]
    return df.select(
        *lst_cols_rest,
        *[col(c).cast(DecimalType(18, n_digitos)).alias(c) for c in lst_cols_decimal],
    ).select(df.columns)


def formatar_cols_float(
    df: SparkDataFrame, lst_cols_float: list
) -> SparkDataFrame:
    """Formatação de colunas passadas como parâmetro para o formato de float. A formatação não altera nem a ordem nem o nome das colunas do dataframe original.

    Parâmetros:
        df (SparkDataFrame): dataframe a ter colunas formatadas para float
        lst_cols_float (list): lista de strings contendo os nomes das colunas a serem formatadas para float. As colunas precisam ser numéricas para que a função funcione

    Retorno:
        SparkDataFrame: dataframe contendo as colunas, cujos nomes foram passados como parâmetro, formatadas para float
    """

    lst_cols_rest = [c for c in df.columns if c not in lst_cols_float]
    return df.select(
        *lst_cols_rest,
        *[col(c).cast(FloatType()).alias(c) for c in lst_cols_float],
    ).select(df.columns)


def formatar_cols_int(
    df: SparkDataFrame, lst_cols_int: list
) -> SparkDataFrame:
    """Formatação de colunas passadas como parâmetro para o formato de inteiro.

    Parâmetros:
        df (SparkDataFrame): dataframe a ter colunas formatadas para inteiro. As colunas precisam ser numéricas para que a função funcione

        lst_cols_int (list): lista de strings contendo os nomes das colunas a serem formatadas para inteiro

    Retorno:
        SparkDataFrame: dataframe contendo as colunas, cujos nomes foram passados como parâmetro, formatadas para inteiro
    """

    lst_cols_rest = [c for c in df.columns if c not in lst_cols_int]
    return df.select(
        *lst_cols_rest, *[col(c).cast(IntegerType()) for c in lst_cols_int]
    )


def formatar_cols_zeros_a_esquerda(
        df: SparkDataFrame, lst_cols_str: list, tam_pad: int
) -> SparkDataFrame:
    """Formatação de colunas passadas como parâmetro para o formato de string de mesmo tamanho, definido pelo parâmetro tam_pad. Para isso, as strings menores são preenchidas com zeros à esquerda.

    Parâmetros:
        df (SparkDataFrame): dataframe a ter colunas formatadas para string contendo zeros à esquerda
        lst_cols_str (list): lista de strings contendo os nomes das colunas a terem zeros preenchidos à esquerda
        tam_pad (int): inteiro que indica o tamanho padronizado das strings. As strings com menos carcateres são preenchidas com zeros à esquerda

    Retorno:
        SparkDataFrame: dataframe contendo as colunas, cujo conteúdo está em formato de string e com tamanho padronizado, possivelmente contendo zeros à esquerda
    """
    cols_df = df.columns
    lst_cols_rest = [c for c in cols_df if c not in lst_cols_str]
    return (df
            .select(*lst_cols_rest, *[lpad(c,tam_pad,'0').alias(c) for c in lst_cols_str])
            .select(cols_df))


def remover_espacos_extra(
    df: SparkDataFrame, lst_nm_cols: list
) -> SparkDataFrame:
    """Essa função remove espaços no início e no final das strings das colunas passadas como parâmetro.

    Parâmetros:
        df (SparkDataFrame): dataframe a ter colunas com espaços extra removidos
        lst_nm_cols (list): lista de strings contendo os nomes das colunas a terem espaços extra no início e no final de suas strings removidos

    Retorno:
        SparkDataFrame: dataframe contendo as colunas sem espaços extra antes e depois das strings nelas contidas
    """
    lst_cols_rest = [c for c in df.columns if c not in lst_nm_cols]
    return df.select(
        *lst_cols_rest, *[trim(col(c)).alias(c) for c in lst_nm_cols]
    )


def agrupar(
    df: SparkDataFrame,
    chv_agrup,
    lst_cols_conta: list = None,
    lst_cols_soma: list = None,
    lst_cols_max: list = None,
    lst_cols_min: list = None,
) -> SparkDataFrame:
    """Agrupa um dataframe pela soma, contagem, máximo ou mínimo de colunas, cujos nomes são passados como parâmetros.



    Parâmetros:
        df (SparkDataFrame): dataframe a ser agrupado

        chv_agrup (_type_): string ou lista de strings contendo o(s) nome(s) da(s) coluna(s) que agrupa(m) o dataframe. Ou seja, que forma(m) a chave de granularidade do dataframe resultate

        lst_cols_conta (list, optional): lista de strings contendo os nomes das colunas a serem contadas no agrupamento. Por padrão, é None. Quando esse parâmetro não é passado, a função não conta colunas ao agrupar o dataframe. Quando é passada, a função conta a quantidade de registros em cada uma das colunas contidas na lista.

        lst_cols_soma (list, optional): lista de strings contendo os nomes das colunas a serem somadas no agrupamento. Por padrão, é None. Quando esse parâmetro não é passado, a função não soma colunas ao agrupar o dataframe. Quando é passada, a função soma o valor dos registros em cada uma das colunas contidas na lista.

        lst_cols_max (list, optional): lista de strings contendo os nomes das colunas a terem seus valores máximos selecionados no agrupamento. Por padrão, é None. Quando esse parâmetro não é passado, a função não seleciona o valor máximo de nenhuma coluna ao agrupar o dataframe. Quando é passada, a função seleciona o valor máximo de cada uma das colunas contidas na lista.

        lst_cols_min (list, optional): lista de strings contendo os nomes das colunas a terem seus valores mínimos selecionados no agrupamento. Por padrão, é None. Quando esse parâmetro não é passado, a função não seleciona o valor mínimo de nenhuma coluna ao agrupar o dataframe. Quando é passada, a função seleciona o valor mínimo de cada uma das colunas contidas na lista.

    Retorno:
        SparkDataFrame: dataframe agrupado pela soma, contagem, máximo e/ou mínimo de colunas passadas como parâmetro
    """

    if lst_cols_conta is None:
        lst_cols_conta = []
    if lst_cols_soma is None:
        lst_cols_soma = []
    if lst_cols_max is None:
        lst_cols_max = []
    if lst_cols_min is None:
        lst_cols_min = []

    lst_cols_conta = [count(c).alias(f"cont_{c}") for c in lst_cols_conta]
    lst_cols_soma = [sparkSum(c).alias(f"soma_{c}") for c in lst_cols_soma]
    lst_cols_max = [sparkMax(c).alias(f"max_{c}") for c in lst_cols_max]
    lst_cols_min = [sparkMin(c).alias(f"min_{c}") for c in lst_cols_min]

    return df.groupBy(chv_agrup).agg(
        *lst_cols_conta, *lst_cols_soma, *lst_cols_max, *lst_cols_min
    )


def mostrar_visao_geral(
    df: SparkDataFrame, nm_df: str, tam_df: int = None
) -> None:
    """Visão das informações gerais do dataframe, que incluem esquema de dados, tamanho da base e visualização dos 5 primeiros registros da base

    Parâmetros:
        df (SparkDataFrame): dataframe a ter suas informações gerais mostradas
        nm_df (str): string contando o nome do dataframe para que seja mostrado nas legendas das informações
        tam_df (int, optional): quantidade de linhas do dataframe. Por default, é None. Quando esse parâmetro é passado, a função melhora em performance.
    """

    if tam_df is None:
        tam_df = df.count()

    print(f"Esquema de dados da base de {nm_df}:")
    df.printSchema()
    print(f"Tamanho da base de {nm_df}:")
    print(f"({tam_df},{len(df.columns)})")
    print(f"\nVisualizacao da base de {nm_df}:")
    df.show(5, truncate=False)

def mostrar_teste_granularidade(
        df: SparkDataFrame, cols_chv, tam_df: int = None
    ) -> None:
    """Teste da granularidade de um dataframe. É printada a quantidade de linhas do dataframe e a quantidade de linhas em que a chave, passada como parâmetro, é distinta. Caso essas duas quantidades sejam iguais, tem-se que a chave passada como parâmetro é, de fato, a granularidade da base. Caso as duas quantidades sejam diferentes, possivelmente há chaves duplicadas. 

    Parâmetros:
        df (SparkDataFrame): dataframe a ter sua granulidade verificada
        cols_chv (_type_): string ou lista de strings contendo colunas que compõem a suposta chave de granularidade da tabela
        tam_df (int, optional): quantidade de linhas do dataframe. 
            Por default, é None. Quando esse parâmetro é passado, a função melhora em performance.
    """
    
    if tam_df is None:
        tam_df = df.count()
    
    print('Tamanho da base:')
    print(tam_df)
    
    qtd_dist = df.select(cols_chv).distinct().count()
    print(f'Quantidade de combinacoes de chave distintas:')
    print(qtd_dist)

    if tam_df == qtd_dist:
        print(f'\n{cols_chv} eh a chave da tabela')
    else:
        print(f'\n{cols_chv} nao eh a chave da tabela')


def orderBy_dict(
    df: SparkDataFrame, d_order: dict
) -> SparkDataFrame:
    """Função responsável por ordenar um dataframe Pyspark baseado em um dicionario.

   Parâmetros:
       df (SparkDataFrame): DataFrame que será ordenado
       d_order (dict): Dicionário com as colunas (chave) e a ordem (valor) do ordenamento.
         ex.: {'coluna1': 'asc', 'coluna2':'desc'}

   Exceções:
       ValueError: Caso algum dos valores do dicionário seja diferente de 'asc' ou 'desc'
         o erro é levantado.

   Retorno:
       _type_: DataFrame ordenado.
   """

    for order in d_order.values():
        if order != "asc" and order != "desc":
            raise ValueError(f"Ordem {order} nao identificada")

    return df.orderBy(
        [
            asc(col(coluna))
            if ordem == "asc"
            else asc(col(coluna))
            if ordem == "desc"
            else None
            for coluna, ordem in d_order.items()
        ]
    )

def obter_top_valores(
    df: SparkDataFrame, nm_col_vlr: str, top_n: int, lst_cols_desc: list
) -> SparkDataFrame:
    """Essa função retorna os primeiros n_top registros, tal que n_top é passado como parâmetro, do dataframe original ordenado de maneira decrescente de acordo com o valor preenchido na coluna numérica, cujo nome é passado como parâmetro. O dataframe retornado possui a coluna de valor e as colunas descritivas, cujos nomes são passados como parâmetro.

    Parâmetros:
        df (SparkDataFrame): DataFrame contendo as colunas de valor e de descrição, cujos nomes são passados como parâmetro
        nm_col_vlr (str): string contendo o nome da coluna de valor a ser utilizada para ordenar o DataFrame
        top_n (int): quantidade de registros do DataFrame a ser retornado
        lst_cols_desc (list): lista de strings contendo os nomes das colunas descritivas a comporem o DataFrame retornado, juntamente com a coluna de valor

    Retorno:
        SparkDataFrame: DataFrame contendo n_top registros, ordenado pela coluna de valor e que contém as colunas descritivas e a coluna de valor
    """
    
    return (df
       .select(*lst_cols_desc,nm_col_vlr)
       .orderBy(desc(nm_col_vlr))
       .limit(top_n))

###############################################
# Valores Ausentes
###############################################


def preencher_nulos(
    df: SparkDataFrame, vlr_a_preencher, nm_cols_a_preencher=None
) -> SparkDataFrame:
    """Substituicao dos registros vazios em uma lista de colunas por um valor pre-determinado.

    Parâmetros:

        df (SparkDataFrame): dataframe cujos valores nulos a serem preenchidos

        vlr_a_preencher (_type_): valor a ser preenchido, que pode ser uma string, um número ou um booleano

        nm_cols_a_preencher (_type_, optional): string ou lista de strings contendo o(s) nome(s) da(s) coluna(s) a serem preenchidas. 
            Por default, é None. Quando esse parâmetro não é passado, a função preenche os nulos de todas as colunas.

    Retorno:

        SparkDataFrame: dataframe com seus valores nulos substituídos por um valor
    """
    if nm_cols_a_preencher is None:
        nm_cols_a_preencher = df.columns

    return df.fillna(vlr_a_preencher, subset=nm_cols_a_preencher)


def obter_qtd_ausentes(
    df: SparkDataFrame, lst_nm_cols: list = None
) -> SparkDataFrame:
    """Calculo da quantidade de registros ausentes de cada uma das colunas, cujos nomes são passados como parâmetro

    Parâmetros:

        df (SparkDataFrame): dataframe que contém a(s) coluna(s) a terem sua(s) quantidade(s) de registros ausentes aferida(s)

        lst_nm_cols (list, optional): lista de strings contendo o(s) nome(s) das colunas a serem aferidas. 
            Por default, é None e faz com que a função retorne a quantidade de registros ausentes para todas as colunas do dataframe

    Retorno:

        SparkDataFrame: dataframe contendo as quantidades de registros ausentes de cada coluna
    """

    if lst_nm_cols is None:
        lst_nm_cols = df.columns

    return df.agg(*[count(when(isnull(c), c)).alias(c) for c in lst_nm_cols])


def obter_pct_ausentes(
    df: SparkDataFrame, lst_nm_cols: list = None, tam_df: int = None
) -> SparkDataFrame:
    """Calculo da porcentagem de registros ausentes de cada uma das colunas, cujos nomes são passados como parâmetro, em relação à quantidade total de linhas do dataframe

    Parâmetros:

        df (SparkDataFrame): dataframe que contém a(s) coluna(s) a terem sua(s) porcentagens(s) de registros ausentes aferida(s)

        lst_nm_cols (list, optional): lista de strings contendo o(s) nome(s) das colunas a serem aferidas. 
            Por default, é None e faz com que a função retorne a porcentagem de registros ausentes para todas as colunas do dataframe

        tam_df (int, optional): quantidade de linhas do dataframe. 
            Por default, é None. Quando esse parâmetro é passado, a função melhora em performance.

    Retorno:

        SparkDataFrame: dataframe contendo as porcentagens de registros ausentes de cada coluna
    """
    if lst_nm_cols is None:
        lst_nm_cols = df.columns

    if tam_df is None:
        tam_df = df.count()

    return df.transform(lambda df: obter_qtd_ausentes(df, lst_nm_cols)).transform(
        lambda df: obter_pct(df, lst_nm_cols, tam_df)
    )


###############################################
# Valores zerados
###############################################


def obter_qtd_zeros(
    df: SparkDataFrame, lst_nm_cols: list
) -> SparkDataFrame:
    """Calculo da quantidade de registros zerados de cada uma das colunas, cujos nomes são passados como parâmetro

    Parâmetros:

        df (SparkDataFrame): dataframe que contém a(s) coluna(s) a terem sua(s) quantidade(s) de registros zerados aferida(s)

        lst_nm_cols (list): lista de strings contendo o(s) nome(s) das colunas a serem aferidas. 
            As colunas precisam ser numéricas para que a função funcione.

    Retorno:

        SparkDataFrame: dataframe contendo as quantidades de zerados de cada coluna
    """
    return df.agg(*[count(when(col(c).eqNullSafe(0), c)).alias(c) for c in lst_nm_cols])


def obter_pct_zeros(
    df: SparkDataFrame, lst_nm_cols: list, tam_df: int = None
) -> SparkDataFrame:
    """Calculo da porcentagem de registros zerados de cada uma das colunas, cujos nomes são passados como parâmetro, em relação à quantidade total de linhas do dataframe.

    Parâmetros:

        df (SparkDataFrame): dataframe que contém a(s) coluna(s) a terem sua(s) porcentagens(s) de registros zerados aferida(s)

        lst_nm_cols (list): lista de strings contendo o(s) nome(s) das colunas a serem aferidas. 
            As colunas precisam ser numéricas para que a função funcione.

        tam_df (int, optional): quantidade de linhas do dataframe. Por default, é None. 
            Quando esse parâmetro é passado, a função melhora em performance.

    Retorno:

        SparkDataFrame: dataframe contendo as porcentagens de registros zerados de cada coluna
    """

    if tam_df is None:
        tam_df = df.count()

    return df.transform(lambda df: obter_qtd_zeros(df, lst_nm_cols)).transform(
        lambda df: obter_pct(df, lst_nm_cols, tam_df)
    )


###############################################
# Valores Distintos
###############################################


def obter_qtd_distintos(
    df: SparkDataFrame, lst_nm_cols: list = None
) -> SparkDataFrame:
    """Calculo da quantidade de registros distintos de cada uma das colunas, cujos nomes são passados como parâmetro

    Parâmetros:

        df (SparkDataFrame): dataframe que contém a(s) coluna(s) a terem sua(s) quantidade(s) de registros distintos aferida(s)

        lst_nm_cols (list, optional): lista de strings contendo o(s) nome(s) das colunas a serem aferidas.
            Por default, é None e faz com que a função retorne a quantidade de registros distintos para todas as colunas do dataframe

    Retorno:

        SparkDataFrame: dataframe contendo as quantidades de distintos de cada coluna
    """
    if lst_nm_cols is None:
        lst_nm_cols = df.columns

    return df.agg(*[countDistinct(col(c)).alias(c) for c in lst_nm_cols])


def obter_pct_distintos(
    df: SparkDataFrame, lst_nm_cols: list = None, tam_df: int = None
) -> SparkDataFrame:
    """Calculo da porcentagem de registros distintos de cada uma das colunas, cujos nomes são passados como parâmetro, em relação à quantidade total de linhas do dataframe.

    Parâmetros:

        df (SparkDataFrame): dataframe que contém a(s) coluna(s) a terem sua(s) porcentagens(s) de registros distintos aferida(s)

        lst_nm_cols (list, optional): lista de strings contendo o(s) nome(s) das colunas a serem aferidas. 
            Por default, é None e faz com que a função retorne a porcentagem de registros distintos para todas as colunas do dataframe.

        tam_df (int, optional): quantidade de linhas do dataframe. 
            Por default, é None. Quando esse parâmetro é passado, a função melhora em performance.

    Retorno:

        SparkDataFrame: dataframe contendo as porcentagens de registros distintos de cada coluna
    """
    if lst_nm_cols is None:
        lst_nm_cols = df.columns

    if tam_df is None:
        tam_df = df.count()

    return df.transform(lambda df: obter_qtd_distintos(df, lst_nm_cols)).transform(
        lambda df: obter_pct(df, lst_nm_cols, tam_df)
    )


def selecionar_cols_distintas(
    df: SparkDataFrame, cols_chave
) -> SparkDataFrame:
    """Calculo dos registros distintos das colunas chave passadas como parâmetro

    Parâmetros:
        df (SparkDataFrame): dataframe que contém as colunas que formarão a chave da tabela de distintos

        cols_chave (_type_): string ou lista de string contendo o(s) nome(s) da(s) coluna(s) a formar a chave de granularidade da tabela

    Retorno:

        SparkDataFrame: dataframe contendo registros distintos das colunas chave
    """
    return df.select(cols_chave).distinct()


###############################################
# Análise Univariada
###############################################


def obter_tab_freq(
    df: SparkDataFrame, nm_col, tam_df: int = None
) -> SparkDataFrame:
    """Essa função retorna uma tabela contendo a quantidade absoluta e relativa de registros em cada uma das categorias ou combinações de categorias

    Parâmetros:

        df (SparkDataFrame): dataframe que contém a coluna ou lista de colunas a terem suas frequências aferidas
        
        nm_col (_type_): string ou lista de strings contendo o nome ou os nomes das colunas a serem aferidas
        
        tam_df (int, optional): quantidade de linhas do dataframe. 
            Por default, é None. Quando esse parâmetro é passado, a função melhora em performance

    Retorno:

        SparkDataFrame: dataframe contendo as frequências da coluna ou lista de colunas
    """
    if tam_df is None:
        tam_df = df.count()

    print("Tabela de frequencias")

    return (
        df.groupBy(nm_col)
        .count()
        .withColumnRenamed("count", "freq_absoluta")
        .withColumn(
            "freq_relativa (%)", sparkRound(col("freq_absoluta") * 100 / tam_df, 2)
        )
        .orderBy(desc("freq_absoluta"))
    )

def obter_tab_freq_periodo(
    df: SparkDataFrame, nm_col_dat: str, periodo: str, tam_df: int = None
) -> SparkDataFrame:
    """Essa função retorna uma tabela contendo a quantidade absoluta e relativa de registros em cada um dos intervalos de tempo de uma coluna de data, cujo nome é passado como parâmetro. A periodicidade dos intervalos de tempo é definida pelo parâmetro de periodo, que pode ser ano ('a'), mês ('m') ou dia ('d'). 

    Parâmetros:
        df (SparkDataFrame): dataframe que contém a coluna de data a ter suas frequências aferidas
        nm_col_dat (str): string contendo o nome da coluna de data a ter suas frequências aferidas. A coluna de data precisa ser do tipo TimeStamp para que função funcione.
        periodo (str): string contendo o nível de agrupamento da coluna de data. Pode assumir os seguintes valores: 'a' para agrupamento por ano, 'm' para agrupamento por mês ou 'd' para agrupamento por dia.
        tam_df (int, optional): quantidade de linhas do dataframe. 
            Por default, é None. Quando esse parâmetro é passado, a função melhora em performance

    Retorno:
        SparkDataFrame: dataframe contendo as frequências da coluna de datas agrupada a um nível definido pelo parâmetro 'periodo'
    """


    if tam_df is None:
        tam_df = df.count()

    funcao_agregamento = {'d':to_date(nm_col_dat),
                          'm':date_format(col(nm_col_dat),"yyyyMM"),
                          'a':year(nm_col_dat)}[periodo] 
    
    periodo = {'d':'data',
               'm':'mes',
               'a':'ano'}[periodo]
    
    return (df
        .groupBy(funcao_agregamento.alias(periodo))
        .count()
        .withColumnRenamed('count','qtd_ocorrencias')
        .withColumn('%',sparkRound(col('qtd_ocorrencias')*100/tam_df,2))
        .orderBy(asc(periodo)))  

###############################################
# Criação de colunas
###############################################

def criar_col_qtd_digitos(
    df: SparkDataFrame, nm_col: str
) -> SparkDataFrame:
    """Essa função retorna o dataframe original, acrescido de uma coluna que informa a quantidade de dígitos dos valores preenchidos em uma coluna numérica, cujo nome é passado como parâmetro

    Parâmetros:
        df (SparkDataFrame): dataframe que contém a coluna numérica a ter a sua quantidade de dígitos aferida
        nm_col (str): string contendo o nome da coluna numérica a ter sua quantidade de dígitos aferida

    Retorno:
        SparkDataFrame: dataframe original acrescido de uma coluna contendo a quantidade de dígitos dos valores preenchidos na coluna numérica, cujo nome é passado como parâmetro
    """
    return df.withColumn('qtd_digitos', length(col(nm_col).cast(StringType())))