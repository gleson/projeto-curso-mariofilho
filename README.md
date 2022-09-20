# Recomendador de Vídeos do Youtube:

Este é um projeto de Machine Learning, desenvolvido em Python que utiliza a biblioteca Streamlit para interação com o usuário ([Clique aqui para acessar](https://recomendador-youtube.herokuapp.com/)). No projeto foi desenvolvido um recomendador de vídeos do Youtube, que classifica os mesmos de acordo com o interesse de visualização, baseado em 5 termos de pesquisa, a saber:
- Data Science
- Machine Learning
- Kaggle
- Streamlit
- Flask

# Cotela de dados:

Os dados foram coletados do Youtube via **Web Scraping** através da biblioteca **beautifulsoup4**, onde foram efetuadas as buscas pelos termos citados e, em cada busca foram selecionados cerca de 410 resultados, com o objetivo de atingir uma base de aproximadamente 2000 registros, tendo em vista que seriam retirados resultados duplicados, com outros assuntos (frutos de teste A/B), ou com dados faltantes como por exemplo, alguns vídeos que apresentavam problemas nas contagens dos Likes como mostra a imagem:

![imagem com bug na contagem dos likes](https://github.com/gleson/projeto-curso-mariofilho/blob/main/bug.jpg)

# Marcando as labels:

Para auxiliar a marcação das labels, desenvolvi um app com o Streamlit, com os botões "Gostei" e "Não Gostei" para facilitar o processo. A ferramenta também exibe as imagens dos vídeos para auxiliar na decisão em títulos que geravam dúvidas. Vide imagem:

![app desenvolvido para auxiliar a marcar as labels como gostei](https://github.com/gleson/projeto-curso-mariofilho/blob/main/marcador.jpg)

Para reduzir a quantidade de itens selecionáveis, foram inseridos filtros, retirando os vídeos com expressões como "Titanic", bem como caracteres chineses, árabes e russos, e também seus devidos canais, pois já tinha certeza que esses eu não iria assistir :P

# Modelagem:

Na modelagem, os melhores resultados foram obtidos com a **RandomForest** e **LightGBM**, e portanto, os mesmos foram selecionados para o **ensemble**, onde este bateu a base line de ambos os modelos, tanto na **Average Precision** como no **RocAUC**. Seguem os valores obtidos:

Ainda durante o curso, o Youtube mudou a estrutura das páginas, quebrando o Web Scraping que seria usado para atualização da base. Esta atualização seria necessária para testes após a modelagem e posteriormente para o deploy. Para evitar a manutenção exaustiva (que perduraria após o deploy), decidi usar a **API do Youtube** para esta tarefa (**Youtube API v3**).

Enquanto desenvolvia as funções para montar o DataFrame obtido pela API, encontrei um erro com vídeos que vinham com a expressão 'P0D' no campo do tempo de vídeo, os itens com esse erro eram vídeos de Lives que estavam agendadas para acontecer, e por esse motivo o Youtube não tinha como informar o tempo dos mesmos, pois ainda não existiam. Criei a rotina para dropar as linhas com essa informação e, aproveitei para exibi-las ao lado, pois poderiam conter algum assunto de interesse ;)

# Conclusão:

O App funcionou perfeitamente, como esperado, recomendando bons vídeos no início. E com a inserção de alguns recursos, o mesmo proporciona ao usuário fazer algumas alterações para uma visualização diferenciada dos resultados, bem como a atualização da base de dados com apenas um clique. Para um projeto inicial, o resultado foi satisfatório ;)

# Agradecimentos:

Agradeço ao **Mário Filho**, pelo excelente curso, com conteúdos muito ricos e densos sobre ML.
