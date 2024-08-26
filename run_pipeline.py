# src/main.py

# Imports 
from substra.sdk.schemas import DatasetSpec, DataSampleSpec, Permissions
import os
import pathlib
import logging










# Setup
from substra import Client

N_CLIENTS = 3

client_0 = Client(client_name="org-1")
client_1 = Client(client_name="org-2")
client_2 = Client(client_name="org-3")

# Create a dictionary to easily access each client from its human-friendly id
clients = {
    client_0.organization_info().organization_id: client_0,
    client_1.organization_info().organization_id: client_1,
    client_2.organization_info().organization_id: client_2,
}

# Store organization IDs
ORGS_ID = list(clients)
ALGO_ORG_ID = ORGS_ID[0]  # Algo provider is defined as the first organization.
DATA_PROVIDER_ORGS_ID = ORGS_ID[1:]  # Data providers orgs are the two last organizations.



# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(f"Number of organizations: {len(ORGS_ID)}")
logger.info(f"Algorithm provider: {ALGO_ORG_ID}")
logger.info(f"Data providers: {DATA_PROVIDER_ORGS_ID}")







# Data and metrics
## Data preparation
import pathlib
import pandas as pd
import pydicom
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the constant for limiting the number of images
N_IMAGES = 20  # You can adjust this value as needed

def setup_vinbigdata(data_path, num_organizations):
    # Load the train.csv file
    train_df = pd.read_csv(data_path / 'train.csv')
    
    # Group by image_id to get all findings for each image
    grouped = train_df.groupby('image_id')
    
    # Prepare data
    images = []
    labels = []
    
    # Get unique image IDs and limit them if necessary
    unique_image_ids = grouped.groups.keys()
    if N_IMAGES and N_IMAGES < len(unique_image_ids):
        unique_image_ids = list(unique_image_ids)[:N_IMAGES]
        logger.info(f"Limiting dataset to {N_IMAGES} images")
    else:
        logger.info(f"Processing all {len(unique_image_ids)} images")
    
    for image_id in unique_image_ids:
        # Load DICOM image
        dicom_path = data_path / 'train' / f'{image_id}.dicom'
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array
        
        # Normalize image
        image = (image - image.min()) / (image.max() - image.min())
        
        # Resize image to 224x224 (common input size for many CNNs)
        image = np.resize(image, (224, 224))
        
        # Get the label (use the most common class_id for this image)
        group = grouped.get_group(image_id)
        label = group['class_id'].mode().iloc[0]
        
        images.append(image)
        labels.append(label)
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    logger.info(f"Processed {len(images)} images")
    
    # Split data for organizations
    splits = np.array_split(range(len(images)), num_organizations)
    
    for i, split in enumerate(splits):
        org_path = data_path / f'org_{i+1}'
        org_path.mkdir(parents=True, exist_ok=True)
        
        # Further split into train and test
        train_idx, test_idx = train_test_split(split, test_size=0.2, random_state=42)
        
        # Save train data
        np.save(org_path / 'train_images.npy', images[train_idx])
        np.save(org_path / 'train_labels.npy', labels[train_idx])
        
        # Save test data
        np.save(org_path / 'test_images.npy', images[test_idx])
        np.save(org_path / 'test_labels.npy', labels[test_idx])
        
        logger.info(f"Organization {i+1}: {len(train_idx)} training images, {len(test_idx)} test images")

# Usage
data_path = pathlib.Path.cwd() / "tmp" / "data_vinbigdata"
setup_vinbigdata(data_path, len(DATA_PROVIDER_ORGS_ID))


print(f"DATA PATH <#################################################3 {data_path}")
print(f"DATA PATH <#################################################3 {data_path}")
print(f"DATA PATH <#################################################3 {data_path}")
print(f"DATA PATH <#################################################3 {data_path}")
print(f"DATA PATH <#################################################3 {data_path}")
print(f"DATA PATH <#################################################3 {data_path}")







## Dataset registration

from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions

assets_directory = pathlib.Path.cwd() / "torch_fedavg_assets"
dataset_keys = {}
train_datasample_keys = {}
test_datasample_keys = {}

for i, org_id in enumerate(DATA_PROVIDER_ORGS_ID):
    client = clients[org_id]

    permissions_dataset = Permissions(public=False, authorized_ids=[ALGO_ORG_ID])

    # Use absolute paths for data_opener and description
    data_opener_path = os.path.abspath(assets_directory / "dataset" / "vinbigdata_opener.py")
    description_path = os.path.abspath(assets_directory / "dataset" / "description.md")

    dataset = DatasetSpec(
        name="VinBigData Chest X-ray",
        data_opener=data_opener_path,
        description=description_path,
        permissions=permissions_dataset,
        logs_permission=permissions_dataset,
    )
    dataset_keys[org_id] = client.add_dataset(dataset)
    assert dataset_keys[org_id], "Missing dataset key"

    # Add the training data on each organization.
    data_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        path=str(data_path / f"org_{i+1}")  # Convert path to string
    )
    train_datasample_keys[org_id] = client.add_data_sample(data_sample)

    # Add the testing data on each organization.
    test_datasample_keys[org_id] = train_datasample_keys[org_id]  # We're using the same data sample for both train and test

logger.info("Data registered successfully.")
logger.info(f"Dataset keys: {dataset_keys}")
logger.info(f"Train datasample keys: {train_datasample_keys}")
logger.info(f"Test datasample keys: {test_datasample_keys}")





## Metrics definition
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

def accuracy(data_from_opener, predictions):
    y_true = data_from_opener["labels"]
    return accuracy_score(y_true, np.argmax(predictions, axis=1))

def roc_auc(data_from_opener, predictions):
    y_true = data_from_opener["labels"]
    n_class = 14  # Number of classes in VinBigData
    y_true_one_hot = np.eye(n_class)[y_true]
    return roc_auc_score(y_true_one_hot, predictions, multi_class='ovr', average='macro')

print(f"Metrics definition. Function accuracy >>>>>>>>>>>>>>>>>>>>>>>> {accuracy}")
print(f"Metrics definition. Function roc_auc >>>>>>>>>>>>>>>>>>>>>>>> {roc_auc}")











## Model definition
import torch
from torch import nn
import torch.nn.functional as F

seed = 42
torch.manual_seed(seed)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 14)  # 14 classes for VinBigData

    def forward(self, x, eval=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=not eval)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()











## Specifying on how much data to train

from substrafl.index_generator import NpIndexGenerator

# Number of model updates between each FL strategy aggregation.
NUM_UPDATES = 100

# Number of samples per update.
BATCH_SIZE = 32

index_generator = NpIndexGenerator(
    batch_size=BATCH_SIZE,
    num_updates=NUM_UPDATES,
)


print(f"Specifying on how much data to train. Complete")





## Torch Dataset definition
class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, data_from_opener, is_inference: bool):
        self.x = data_from_opener["images"]
        self.y = data_from_opener["labels"]
        self.is_inference = is_inference

    def __getitem__(self, idx):
        if self.is_inference:
            x = torch.FloatTensor(self.x[idx][None, ...])
            return x
        else:
            x = torch.FloatTensor(self.x[idx][None, ...])
            y = torch.tensor(self.y[idx]).type(torch.int64)
            y = F.one_hot(y, 14)  # 14 classes for VinBigData
            y = y.type(torch.float32)
            return x, y

    def __len__(self):
        return len(self.x)


print(f"Torch Dataset definition. Complete")















## SubstraFL algo definition

from substrafl.algorithms.pytorch import TorchFedAvgAlgo


class TorchCNN(TorchFedAvgAlgo):
    def __init__(self):
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            index_generator=index_generator,
            dataset=TorchDataset,
            seed=seed,
            use_gpu=False,
        )

print(f"SubstraFL algo definition. Complete")











## Federated Learning strategies
from substrafl.strategies import FedAvg

strategy = FedAvg(algo=TorchCNN(), metric_functions={"Accuracy": accuracy, "ROC AUC": roc_auc})


print(f"Federated Learning strategies. Complete")














## Where to train where to aggregate
from substrafl.nodes import TrainDataNode
from substrafl.nodes import AggregationNode


aggregation_node = AggregationNode(ALGO_ORG_ID)

# Create the Train Data Nodes (or training tasks) and save them in a list
train_data_nodes = [
    TrainDataNode(
        organization_id=org_id,
        data_manager_key=dataset_keys[org_id],
        data_sample_keys=[train_datasample_keys[org_id]],
    )
    for org_id in DATA_PROVIDER_ORGS_ID
]

print(f"Where to train where to aggregate. Complete")













## Where and when to test
from substrafl.nodes import TestDataNode
from substrafl.evaluation_strategy import EvaluationStrategy

# Create the Test Data Nodes (or testing tasks) and save them in a list
test_data_nodes = [
    TestDataNode(
        organization_id=org_id,
        data_manager_key=dataset_keys[org_id],
        data_sample_keys=[test_datasample_keys[org_id]],
    )
    for org_id in DATA_PROVIDER_ORGS_ID
]


# Test at the end of every round
my_eval_strategy = EvaluationStrategy(test_data_nodes=test_data_nodes, eval_frequency=1)


print(f"Where and when to test. Complete")





# Running the experiment
## specify the third parties dependencies required to run it
from substrafl.dependency import Dependency

dependencies = Dependency(
    pypi_dependencies=[
        "numpy==1.24.3",
        "scikit-learn==1.3.1",
        "torch==2.0.1",
        "pydicom==2.3.1",
        "pandas==1.5.3",
        "--extra-index-url https://download.pytorch.org/whl/cpu"
    ]
)
print(f"specify the third parties dependencies required to run it. Complete")




## execute_experiment
from substrafl.experiment import execute_experiment
import logging
import substrafl

substrafl.set_logging_level(loglevel=logging.ERROR)
# A round is defined by a local training step followed by an aggregation operation
NUM_ROUNDS = 10

print(f"execute_experiment. Started ")


compute_plan = execute_experiment(
    client=clients[ALGO_ORG_ID],
    strategy=strategy,
    train_data_nodes=train_data_nodes,
    evaluation_strategy=my_eval_strategy,
    aggregation_node=aggregation_node,
    num_rounds=NUM_ROUNDS,
    experiment_folder=str(pathlib.Path.cwd() / "tmp" / "experiment_summaries"),
    dependencies=dependencies,
    clean_models=False,
    name="MNIST documentation example",
)


print(f"execute_experiment. Complete  ")


# The results will be available once the compute plan is completed
client_0.wait_compute_plan(compute_plan.key)






## List results
import pandas as pd

performances_df = pd.DataFrame(client.get_performances(compute_plan.key).model_dump())
print("\nPerformance Table: \n")
print(performances_df[["worker", "round_idx", "identifier", "performance"]])

print(f"List results. Complete  ")







## Plot results
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Test dataset results")

axs[0].set_title("Accuracy")
axs[1].set_title("ROC AUC")

for ax in axs.flat:
    ax.set(xlabel="Rounds", ylabel="Score")


for org_id in DATA_PROVIDER_ORGS_ID:
    org_df = performances_df[performances_df["worker"] == org_id]
    acc_df = org_df[org_df["identifier"] == "Accuracy"]
    axs[0].plot(acc_df["round_idx"], acc_df["performance"], label=org_id)

    auc_df = org_df[org_df["identifier"] == "ROC AUC"]
    axs[1].plot(auc_df["round_idx"], auc_df["performance"], label=org_id)

plt.legend(loc="lower right")
plt.show()

print(f"Plot results. Complete  ")


## Download a model
from substrafl.model_loading import download_algo_state

client_to_download_from = DATA_PROVIDER_ORGS_ID[0]
round_idx = None

algo = download_algo_state(
    client=clients[client_to_download_from],
    compute_plan_key=compute_plan.key,
    round_idx=round_idx,
)

model = algo.model

print(model)
print(f"Download a model. Complete  ")
