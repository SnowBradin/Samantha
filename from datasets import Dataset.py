from datasets import Dataset
# Carregue o conjunto de dados "nome-do-dataset"
dataset = Dataset.load("nome-do-dataset")
from datasets import Dataset
# Carregue o conjunto de dados local "caminho/para/o/dataset"
dataset = Dataset.from_file("caminho/para/o/dataset")
# Filtre as amostras onde a coluna "coluna_name" é igual a "valor"
filtered_dataset = dataset.filter(lambda x: x["coluna_name"] == "valor")

# Selecione as primeiras 10 amostras
selected_dataset = dataset.select(range(10))
from datasets import DatasetDict

# Divida o dataset em 80% para treino, 10% para validação e 10% para teste
dataset = dataset.train_test_split(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

# Acesse os datasets de treino, validação e teste
train_dataset = dataset["train"]
val_dataset = dataset["val"]
test_dataset = dataset["test"]