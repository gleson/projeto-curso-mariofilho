import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
from datetime import datetime
from scipy.sparse import hstack
from googleapiclient.discovery import build
import streamlit.components.v1 as components

today = datetime.now().date()


queries = st.sidebar.multiselect(
		'Selecione os termos de busca que serão exibidos:',
		['Machine Learning', 'Data Science', 'Kaggle', 'Flask', 'Streamlit'],
		['Machine Learning', 'Data Science', 'Kaggle', 'Flask', 'Streamlit'])


qty = st.sidebar.slider('Quantidade de resultados exibidos:', 15, 80, 35)


# calculando time_sec
def calc_sec(text):
	x = text.replace("PT","").replace("S","").replace("M"," ").replace("H"," ").split(" ")
	return (int(x[-3])*3600 if len(x)>2 else 0) + (int(x[-2])*60  if len(x)>1 else 0) + (int(x[-1])  if x[-1]!="" else 0)


df = pd.DataFrame()


col1, col2, col3 = st.columns(3)
with col1:
	refresh = st.button('🔄 Atualizar (via API)')
with col2:
	option = st.selectbox(
		'Escolha uma visualização:',
		('Melhor Visual', 'Básica', 'DataFrame'))
with col3:
	know_more = st.checkbox('Este projeto foi desenv... Saiba mais')



st.write("## Recomendador de vídeos do Youtube")
st.markdown("***")


if refresh:
	qty_api = int(120/len(queries))

	class KeyList:
		def __init__(self, listing):
			self.listing = listing

	klist = jb.load("klist.pkl.z")
	youtube=build('youtube', 'v3', developerKey=("y".join(klist.listing[:3])+"W".join(klist.listing[3:6])))
	
	for query in queries:
		search = youtube.search().list(q=query, part="id,snippet", maxResults=qty_api, regionCode='BR').execute()

		# Obtendo os valores das features
		for i in range(qty_api):
			try:
				video_id = search['items'][i]['id']['videoId']
				response = youtube.videos().list(part='statistics, contentDetails, topicDetails', id=video_id).execute()
	
				df = df.append({
					'id': search['items'][i]['id']['videoId'],
					'title': search['items'][i]['snippet']['title'],
					'channel': search['items'][i]['snippet']['channelTitle'],
					'published': search['items'][i]['snippet']['publishedAt'],
					'query': query,
					'time_sec': response['items'][0]['contentDetails']['duration'],
					'views': response['items'][0]['statistics']['viewCount'],
					'likes': response['items'][0]['statistics']['likeCount'],
					'dislikes': 0
					# 'dislikes': response['items'][0]['statistics']['dislikeCount']
				}, ignore_index=True)
			except:
				continue

	df.drop_duplicates(subset=['id'], inplace=True)
	df.to_csv('df_api.csv', index=False)

	try:
		df_lives = df[df.time_sec == 'P0D']
		if df_lives.shape[0] != 0:
			df_lives.to_csv('df_lives.csv', index=False)
	except:
		pass

df = pd.read_csv("df_api.csv")

df_lives = pd.read_csv("df_lives.csv")


st.sidebar.markdown("***")

colsb = st.sidebar.columns(1)
if df_lives.shape[0] != 0:
	st.sidebar.write('### Lives agendadas ou recentes:')
	for index, row in df_lives.iterrows():
		st.sidebar.write(f'[{row.title}](https://www.youtube.com/watch?v={row.id})')
		published_at = row.published.replace("T"," ").split("Z")
		st.sidebar.text(f'-> Quando: {published_at[0]}')
		


st.sidebar.write("### Quem sou:")
expander = st.sidebar.expander(label='Saiba mais')
expander.write("Me chamo Gleson Cruz.")
expander.write("Sou Profissional de TI e desenvolvedor Python/Django, formado como Tecnólogo em Redes de Computadores, moro em Fortaleza/CE.")
expander.write("Possuo habilidades com as linguagens: Python, PHP, MySQL, HTML5, CSS3, JavaScript, VBScript, Batch, VBA e tenho conhecimento em Ciência de Dados.")
expander.write("Conheça o meu Linkedin: https://www.linkedin.com/in/gleson-cruz/")

try:
	df = df[df.time_sec != 'P0D'] # Retirando as lives agendadas do DataFrame principal 
except:
	pass

df = df.astype({'views': int, 'likes': int, 'dislikes': int})


# gerando as features
df['days'] = [ (pd.to_datetime(today) - pd.to_datetime(published[:10]))//np.timedelta64(1, 'D') for published in df['published'] ]
df.days.replace(0, 1, inplace=True)

# calculando time_sec, daily_views, daily_likes e daily_dislikes
df['time_sec'] = [calc_sec(x) for x in df['time_sec']]
df['daily_views'] = round(df['views']/df['days'], 2)
df['daily_likes'] = round(df['likes']/df['days'], 2)
df['daily_dislikes'] = round(df['dislikes']/df['days'], 2)

# Dropando as features desnecessárias
df.drop(['published', 'views', 'likes', 'dislikes', 'days'], axis=1, inplace=True)


# Selecionando grupos de features
numeric_features = df[['time_sec', 'daily_views', 'daily_likes', 'daily_dislikes']]
ohe_features = ['channel', 'query']


#carregando modelos
mdl_rf	= jb.load("rf.pkl.z")
mdl_lgbm  = jb.load("lgbm.pkl.z")
title_vec = jb.load('title_vec.pkl.z')
ohe_ct	= jb.load("columnTransformer.pkl.z")


# Transformando o título
title_bow = title_vec.transform(df['title'])
# Transformando campos do OHE
ohe_bow = ohe_ct.transform(df[ohe_features])


# DataFrame Final
final_df = hstack([numeric_features, title_bow, ohe_bow])


# Probs para o ensemble
pp_rf = mdl_rf.predict_proba(final_df)[:, 1]
pp_lgbm = mdl_lgbm.predict_proba(final_df)[:, 1]

pp = pp_rf*0.5 + pp_lgbm*0.5


df_p = df[['id', 'title', 'channel', 'query', 'time_sec']].copy()
df_p['p'] = pp



df_p.sort_values(by='p', ascending=False, inplace=True)

df_p = df_p[['id', 'title', 'channel', 'query', 'p']][df_p['query'].isin(queries)][:qty]



if know_more == False:

	if len(queries) == 0:
		st.write("# Favor selecione os termos para exibição")
	else:
		i = 1
		if option == 'Melhor Visual':
			for index, row in df_p.iterrows():
				components.html(f"""
				<table width="100%">
				<tr style="border-bottom: 1px solid silver;">
					<td style="width: 35%; padding-right: 10px;"><img src="https://i.ytimg.com/vi/{row.id}/mqdefault.jpg" style="max-width: 100%; height: 100%;"></td>
					<td width="65%">
						<p><b>{i}# <a href="https://www.youtube.com/watch?v={row.id}">{row.title}</a></b></p>
						<div style="width: 100%;">
							<div style="float: left; width: 50%;"><p><b>{row.channel}</b></p><p>query: {row.query}</p></div>
							<div style="float: left; width: 50%; font-size: 1.5em;"><p><b>p: {round(row.p*100,2)}%</b></p></div>
						</div>
					</td>
				</tr>
				</table>
				""")
				i += 1
		elif option == 'Básica':
			for index, row in df_p[:qty].iterrows():
				st.write(f'**{round(row.p*100,2)}%** # [{row.title}](https://www.youtube.com/watch?v={row.id})')
				i += 1
		else:
			st.write(df_p)

else:
	st.write("### Projeto:")
	"""
	Este projeto foi desenvolvido em Python com a biblioteca Streamlit e trata-se de um recomendador de vídeos do Youtube, que classifica os mesmos de acordo com o interesse de visualização, baseado em 5 termos de pesquisa, a saber:
	- Data Science
	- Machine Learning
	- Kaggle
	- Streamlit
	- Flask
	"""
	st.write("### Cotela de dados:")
	"""
	Os dados iniciais foram coletados no Youtube via Web Scraping através da biblioteca beautifulsoup4, onde foram efetuadas as buscas pelos termos citados e, em cada busca foram 
	selecionados cerca de 410 resultados, com o objetivo de atingir uma base de aproximadamente 2000 registros, tendo em vista que seriam retirados resultados duplicados, 
	resultados com outros assuntos (frutos de teste A/B), ou com dados faltantes, como por exemplo, alguns vídeos que apresentavam problemas nas contagens dos Likes como mostra a imagem:
	"""
	st.image('bug.jpg', caption='(Vídeo sem a contagem de Likes. Pode ser acessado através do link: https://www.youtube.com/watch?v=dlUXr0mFEhA)')

	st.write("### Marcando as labels:")
	"""
	Para auxiliar a marcação das labels, desenvolvi um app com o Streamlit, com os botões "Gostei" e "Não Gostei" para facilitar o processo. A ferramenta também exibia a imagem do vídeo (que ajudava na decisão em títulos que geravam dúvida). Vide imagem:
	"""
	st.image('marcador.jpg', caption='App desenvolvido para a marcação das labels.')
	"""
	Para reduzir a quantidade de itens selecionáveis, foram inseridos filtros, retirando os vídeos com expressões como "Titanic", bem como caracteres chineses, árabes e russos, e também seus devidos canais, pois esses, eu já tinha certeza que não iria assistir :P
	"""
	st.write("### Modelagem:")
	"""
	Na modelagem, os melhores resultados foram obtidos com a RandomForest e LightGBM, e portanto, os mesmos foram selecionados para o ensemble, onde este bateu a base line de ambos os modelos, tanto na Average Precision como no RocAUC. Seguem os valores obtidos:
	"""
	df_ens = pd.DataFrame({'Model': ['RF', 'LGBM', 'Ensemble'], 'AvrgPrec': [0.29175, 0.29632, 0.30603], 'RocAuc': [0.71131, 0.70872, 0.71801]})
	st.write(df_ens)
	st.write("### Atualização da base:")
	"""
	Ainda durante o curso, o Youtube mudou a estrutura das páginas, quebrando o Web Scraping que seria usado para atualização da base. Esta atualização seria necessária para testes após a modelagem e posteriormente para o deploy. Para evitar a manutenção exaustiva (que perduraria após o deploy), decidi usar a API do Youtube para esta tarefa (Youtube API v3).

	Enquanto desenvolvia as funções para montar o DataFrame obtido pela API, encontrei um erro com vídeos que vinham com a expressão 'P0D' no campo do tempo de vídeo, os itens com esse erro eram vídeos de Lives que estavam agendadas para acontecer, e por esse motivo o Youtube não tinha como informar o tempo dos mesmos, pois ainda não existiam. Criei a rotina para dropar as linhas com essa informação e, aproveitei para exibi-las ao lado, pois poderiam conter algum assunto de interesse ;)
	"""
	st.write("### Conclusão:")
	"""
	O App funcionou perfeitamente como esperado, recomendando bons vídeos no início. E com a inserção de alguns recursos, o mesmo proporciona ao usuário fazer algumas alterações para uma visualização diferenciada dos resultados. Bem como a atualização da base de dados com apenas um clique. Para um projeto inicial, o resultado está satisfatório ;)
	"""
	st.write("### Agradecimentos:")
	"""
	Agradeço ao **Mário Filho** pelo excelente curso, com conteúdos bastante ricos e diferenciados sobre ML.
	"""
