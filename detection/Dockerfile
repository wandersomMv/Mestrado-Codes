# Imagem base
FROM python:3.9


# Instalar pacotes necessários
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Definir diretório de trabalho
WORKDIR /app

# Copiar os arquivos de código-fonte para o diretório de trabalho
COPY . /app

# Instalar as dependências
RUN pip install  -r requirements.txt

# Exponha a porta que sua aplicação Flask está ouvindo
EXPOSE 5000

# Comando para iniciar o servidor da API
CMD [ "python", "app.py" ]

