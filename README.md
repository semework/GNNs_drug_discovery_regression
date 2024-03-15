# GNNs_drug_discovery_regression &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 
<div style="text-align:center;color:Blue">
    <h2> Graph Regression with Graph Neural Networks</h2>
</div>

### Several studies have shown that Graph neural networks (GNNs) have a greater potential for drug discovery efforts compared to traditional predictive/descriptor approaches. They have been highly useful as modelling tools for molecular property studies, such as solubility and binding. 

### *Here, apply graph regression to assign one y-value to an entire graph (in contrast to nodes).* 
1. We start with a dataset of graphs, based on some structural graph properties - in today's case based on lipophilicity.
2. Accordingly, entire graphs are embedded in such a way that helps us predict a molecular property prediction (a single 
lipophilicity value) for each. One can use these embeddings further to do more analysis, such as 
to classify them based on a value, such as a lipophilicity range.

### About this repo   
#### *A jupyter notebook (```GNN_regression_lipophilicity.ipynb```) which performs the above procedures and provides figures and tables:*

#### Packages needed:
- math
- matplotlib
- networkx
- numpy
- os
- pandas
- pubchempy
- rdkit
- sklearn
- torch
- torch_geometric

#### This notebook presents a thorough approach on how to apply Graph Neural Networks (GNNs) to solve a graph regression problem. 

We apply graph regression to assign one y-value to an entire graph (in contrast to nodes). 
1. We start with a dataset of graphs, based on some structural graph properties - in today's case based on lipophilicity.
2. Accordingly, entire graphs are embedded in such a way that helps us predict a molecular property prediction (a single 
lipophilicity value) for each. One can use these embeddings further to do more analysis, such as 
to classify them based on a value, such as a lipophilicity range.

Play with the model settings, data splitting, training setups etc. to get the best results from this code and any other data/model you implement based on the information provided.

#### The data: Experimental results of octanol/water distribution coefficient (logD at pH 7.4). 
Lipophilicity is an important feature of drug molecules that affects both membrane permeability 
and solubility - thus a molecule's interactivity with other molecules. 

1.	Import lipophilicity data for 4000 molecules (https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv, or use Python)
2. from PyTorch Geometric’s dataset library (https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html)
   
This Lipophilicity dataset is curated from ChEMBL database, provides experimental results of 
octanol/water distribution coefficient (logD at pH 7.4) of 4200 compounds. 
Read more: https://arxiv.org/pdf/1703.00564.pdf


```python
from pandas.plotting import table
from rdkit.Chem import Draw
from rdkit import Chem
from sklearn.metrics import r2_score
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import to_networkx
from torch.nn import Linear
  
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import pubchempy
import rdkit
import time
import torch
import torch.nn.functional as F 
import warnings
warnings.filterwarnings("ignore")
```

<div style="text-align:center;color:Blue">
    <h3> Lipophilicty dataset import (load if already imported) and explore</h3>
</div>


```python
dataset = MoleculeNet(root=".", name="lipo")
data = dataset[0]
```


```python
print('\n======== Dataset =======\n')
print("Dataset type: ", type(dataset))
print("Dataset size (graphs): ", len(dataset))
print("Dataset features: ", dataset.num_features)
print("Dataset target: ", dataset.num_classes)
print("Dataset length: ", dataset.len)
print('\n======== first sample =======\n')
print("Dataset sample: ", data)
print("Sample  nodes: ", data.num_nodes)
print("Sample  edges: ", data.num_edges)
```

    
    ======== Dataset =======
    
    Dataset type:  <class 'torch_geometric.datasets.molecule_net.MoleculeNet'>
    Dataset size (graphs):  4200
    Dataset features:  9
    Dataset target:  553
    Dataset length:  <bound method InMemoryDataset.len of Lipophilicity(4200)>
    
    ======== first sample =======
    
    Dataset sample:  Data(x=[26, 9], edge_index=[2, 56], edge_attr=[56, 3], smiles='C[C@H](Nc1nc(Nc2cc(C)[nH]n2)c(C)nc1C#N)c3ccc(F)cn3', y=[1, 1])
    Sample  nodes:  26
    Sample  edges:  56


 <div style="text-align:left;color:Maroon">
    <h4> Take a look at the first 5 nodes from the first sample molecule</h4>
</div>


```python
dataset[0].x[:5]
```




    tensor([[6, 0, 4, 5, 3, 0, 4, 0, 0],
            [7, 0, 3, 5, 0, 0, 3, 1, 1],
            [6, 0, 3, 5, 0, 0, 3, 1, 1],
            [6, 0, 4, 5, 2, 0, 4, 0, 0],
            [7, 0, 3, 5, 0, 0, 4, 0, 1]])



<div style="text-align:left;color:Maroon">
    <h4> The first 5 sparse matrices (COO)</h4>
</div>


```python
dataset[0].edge_index.t()[:5]
```




    tensor([[ 0,  1],
            [ 1,  0],
            [ 1,  2],
            [ 1, 23],
            [ 2,  1]])



<div style="text-align:left;color:Maroon">
    <h4> The target (lipophilicty value for the first data point (i.e. molecule))</h4>
</div>


```python
dataset[0].y
```




    tensor([[3.5400]])



<div style="text-align:left;color:Maroon">
    <h4>Use pubchempy to get the name and rdkit to draw the first molecular structure</h4>
</div>


```python
sm = dataset[0]['smiles']
compound = pubchempy.get_compounds(sm, namespace='smiles')
match = compound[0]
match.iupac_name
```




    '2-[[4-(4-chlorophenyl)piperazin-1-yl]methyl]-1-methylbenzimidazole'




```python
molecule = Draw.MolsToGridImage([Chem.MolFromSmiles(sm)], 
                molsPerRow=1, subImgSize=(300,200), returnPNG=False)
molecule.save('assets/images/first_molecule.png')
molecule
```




    
![png](assets/images/first_molecule.png)
    



<div style="text-align:left;color:Maroon">
    <h4>Let's build a dataframe with some important attributes for each of the first 12 molecules. Twelve because it makes grid plotting easier. Here, we convert a tensor (the target, data.y) to an array for simplicity.</h4>
</div>


```python
top_n = 12
data_attrib = []

for data in dataset[:top_n]:
    data_attrib.append([data.num_nodes, data.num_edges, data.smiles, np.array(data.y[0])[0]])
    
data_attr = pd.DataFrame(data_attrib)
data_attr.columns = ['num_nodes', 'num_edges', 'smiles', 'target_logD']
data_attr.head(top_n)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_nodes</th>
      <th>num_edges</th>
      <th>smiles</th>
      <th>target_logD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24</td>
      <td>54</td>
      <td>Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14</td>
      <td>3.54</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>70</td>
      <td>COc1cc(OC)c(cc1NC(=O)CSCC(=O)O)S(=O)(=O)N2C(C)...</td>
      <td>-1.18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21</td>
      <td>46</td>
      <td>COC(=O)[C@@H](N1CCc2sccc2C1)c3ccccc3Cl</td>
      <td>3.69</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28</td>
      <td>62</td>
      <td>OC[C@H](O)CN1C(=O)C(Cc2ccccc12)NC(=O)c3cc4cc(C...</td>
      <td>3.37</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>58</td>
      <td>Cc1cccc(C[C@H](NC(=O)c2cc(nn2C)C(C)(C)C)C(=O)N...</td>
      <td>3.10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>23</td>
      <td>52</td>
      <td>OC1(CN2CCC1CC2)C#Cc3ccc(cc3)c4ccccc4</td>
      <td>3.14</td>
    </tr>
    <tr>
      <th>6</th>
      <td>35</td>
      <td>74</td>
      <td>COc1cc(OC)c(cc1NC(=O)CCC(=O)O)S(=O)(=O)NCc2ccc...</td>
      <td>-0.72</td>
    </tr>
    <tr>
      <th>7</th>
      <td>34</td>
      <td>72</td>
      <td>CNc1cccc(CCOc2ccc(C[C@H](NC(=O)c3c(Cl)cccc3Cl)...</td>
      <td>0.34</td>
    </tr>
    <tr>
      <th>8</th>
      <td>23</td>
      <td>50</td>
      <td>COc1ccc(cc1)C2=COc3cc(OC)cc(OC)c3C2=O</td>
      <td>3.05</td>
    </tr>
    <tr>
      <th>9</th>
      <td>15</td>
      <td>34</td>
      <td>Oc1ncnc2scc(c3ccsc3)c12</td>
      <td>2.25</td>
    </tr>
    <tr>
      <th>10</th>
      <td>28</td>
      <td>62</td>
      <td>CS(=O)(=O)c1ccc(Oc2ccc(cc2)C#C[C@]3(O)CN4CCC3C...</td>
      <td>1.51</td>
    </tr>
    <tr>
      <th>11</th>
      <td>26</td>
      <td>56</td>
      <td>C[C@H](Nc1nc(Nc2cc(C)[nH]n2)c(C)nc1C#N)c3ccc(F...</td>
      <td>2.61</td>
    </tr>
  </tbody>
</table>
</div>



<div style="text-align:left;color:Maroon">
    <h4>Next, we plot the molecular structure for the first 12 molecules. We use the package 'pubchempy' to convert smiles into chemical names. Then we shorten the names for visual clarity (in subplot titles and the above dataframe). To do so, we split the name and use the last string piece after the last ')' or ']', whichever is the shortest.</h4>
</div>


```python
ch_names = []
for i in data_attr["smiles"]:
    compounds = pubchempy.get_compounds(i, namespace='smiles')
    match = compounds[0]
    names = [match.iupac_name.split(')')[-1], match.iupac_name.split(']')[-1]]
    res = min(names, key=len)
    ch_names.append(res)
chem_names = ['... ' + str(i) for i in ch_names]
data_attr["name"] = chem_names
data_attr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_nodes</th>
      <th>num_edges</th>
      <th>smiles</th>
      <th>target_logD</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24</td>
      <td>54</td>
      <td>Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14</td>
      <td>3.54</td>
      <td>... -1-methylbenzimidazole</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>70</td>
      <td>COc1cc(OC)c(cc1NC(=O)CSCC(=O)O)S(=O)(=O)N2C(C)...</td>
      <td>-1.18</td>
      <td>... sulfanylacetic acid</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21</td>
      <td>46</td>
      <td>COC(=O)[C@@H](N1CCc2sccc2C1)c3ccccc3Cl</td>
      <td>3.69</td>
      <td>... acetate</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28</td>
      <td>62</td>
      <td>OC[C@H](O)CN1C(=O)C(Cc2ccccc12)NC(=O)c3cc4cc(C...</td>
      <td>3.37</td>
      <td>... pyrrole-5-carboxamide</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>58</td>
      <td>Cc1cccc(C[C@H](NC(=O)c2cc(nn2C)C(C)(C)C)C(=O)N...</td>
      <td>3.10</td>
      <td>... -2-methylpyrazole-3-carboxamide</td>
    </tr>
    <tr>
      <th>5</th>
      <td>23</td>
      <td>52</td>
      <td>OC1(CN2CCC1CC2)C#Cc3ccc(cc3)c4ccccc4</td>
      <td>3.14</td>
      <td>... octan-3-ol</td>
    </tr>
    <tr>
      <th>6</th>
      <td>35</td>
      <td>74</td>
      <td>COc1cc(OC)c(cc1NC(=O)CCC(=O)O)S(=O)(=O)NCc2ccc...</td>
      <td>-0.72</td>
      <td>... -4-oxobutanoic acid</td>
    </tr>
    <tr>
      <th>7</th>
      <td>34</td>
      <td>72</td>
      <td>CNc1cccc(CCOc2ccc(C[C@H](NC(=O)c3c(Cl)cccc3Cl)...</td>
      <td>0.34</td>
      <td>... propanoic acid</td>
    </tr>
    <tr>
      <th>8</th>
      <td>23</td>
      <td>50</td>
      <td>COc1ccc(cc1)C2=COc3cc(OC)cc(OC)c3C2=O</td>
      <td>3.05</td>
      <td>... chromen-4-one</td>
    </tr>
    <tr>
      <th>9</th>
      <td>15</td>
      <td>34</td>
      <td>Oc1ncnc2scc(c3ccsc3)c12</td>
      <td>2.25</td>
      <td>... pyrimidin-4-one</td>
    </tr>
    <tr>
      <th>10</th>
      <td>28</td>
      <td>62</td>
      <td>CS(=O)(=O)c1ccc(Oc2ccc(cc2)C#C[C@]3(O)CN4CCC3C...</td>
      <td>1.51</td>
      <td>... octan-3-ol</td>
    </tr>
    <tr>
      <th>11</th>
      <td>26</td>
      <td>56</td>
      <td>C[C@H](Nc1nc(Nc2cc(C)[nH]n2)c(C)nc1C#N)c3ccc(F...</td>
      <td>2.61</td>
      <td>... pyrazine-2-carbonitrile</td>
    </tr>
  </tbody>
</table>
</div>



<div style="text-align:left;color:Maroon">
    <h4>Save the dataframe as an image using a great python package called 'dataframe_image'</h4>
</div>


```python
import dataframe_image as dfi
 
df_styled =  data_attr.style.background_gradient() 
dfi.export(df_styled, "assets/images/data_attrib.png")
```

<div style="text-align:left;color:Maroon">
    <h4>We are now ready to plot the 12 molecules</h4>
</div>


```python
img = Draw.MolsToGridImage([Chem.MolFromSmiles(data_attr["smiles"][i]) for i in range(top_n)], 
                         molsPerRow=4,subImgSize=(300,200), legends=list(data_attr.name.values),
                         returnPNG=False)
img.save('assets/images/molecules.png')
img
```




    
![png](assets/images/molecules.png)
    



<div style="text-align:left;color:Maroon">
    <h4>To visualize two of the graphs (the first and tenth graph from our original dataset), we convert PyTorch Geometric graph to to NetworkX. Note that there are 24 nodes in the first graph and 27 in the second.</h4>
</div>


```python
import matplotlib.pyplot as plt

def visualize_net():
    plt.figure(figsize=(11,5))
    ax1 = plt.subplot2grid(shape=(1, 2), loc=(0,0))
    ax2 = plt.subplot2grid(shape=(1, 2), loc=(0,1))
    ax1.set_title('First graph nodes')
    ax2.set_title('Tenth graph nodes')

    G = to_networkx(dataset[0], to_undirected=True)
    
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=14), with_labels=True,
                     node_color=[0.91, 0.91, 0.95], node_size=400, width=1.2,
                     edgecolors='blue', cmap="Set2", ax=ax1)
    
    G = to_networkx(dataset[10], to_undirected=True)
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=14), with_labels=True,
                    node_color=[0.91, 0.91, 0.95], node_size=400, width=1.2,
                    edgecolors='blue', cmap="Set2", ax=ax2)
    plt.tight_layout()
    plt.savefig('assets/images/graphs.png', bbox_inches='tight')
    plt.show()    
    
visualize_net()
```


    
![png](assets/images/graphs.png)
    


<div style="text-align:center;color:Blue">
    <h3> Solubility regression with GNN</h3>
</div>

#### Steps: 
1. Create a GCN model structure that contains three GCNConv layers, and 64 hidden channels.  

2. Perform graph level (one y-value per graph) prediction


```python
embedding_size = 64
class GCN(torch.nn.Module):
    def __init__(self):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(data.num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        # Output layer
        self.out = Linear(embedding_size*2, 1)

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)

        # Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)
          
        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)

        # Classifier (Linear).
        out = self.out(hidden)

        return out, hidden

model = GCN()
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

```

    GCN(
      (initial_conv): GCNConv(9, 64)
      (conv1): GCNConv(64, 64)
      (conv2): GCNConv(64, 64)
      (conv3): GCNConv(64, 64)
      (out): Linear(in_features=128, out_features=1, bias=True)
    )
    Number of parameters:  13249


<div style="text-align:center;color:Blue">
    <h3> Train the GNN</h3>
</div>

Here, we use sklearn's r2_score to measure performance to follow accuracy through time. As this is a regression problem, the right metric is RMSE, but for visual follow-up we calculate accuracy as it is the most intuitive metric. Training using 500 epochs takes about 11 minutes on a Macbook Pro with 64 GB 2667 MHz DDR4, 2.4 GHz 8-Core Intel Core i9, AMD Radeon Pro 5600M 8 GB, Intel UHD Graphics 630 1536 MB. 

Automated batching multiple graphs into a single giant graph is taken care of by PyTorch Geometric's torch_geometric.data.DataLoader class.

#### Training consists of these three major steps:

1. Embed
2. Aggregate into a readout graph
3. Use a function to convert the readout into a classifier.

Depending on how long the model is trained, accuracy can reach as high as 99%, which is undesirable (overtraining). As the dataset is very small, you will notice accuracy and loss fluctuations.
<br>


```python
from torch_geometric.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

# Root mean squared error
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)  

# Calculate accuracy r2
def r2_accuracy(pred_y, y):
    score = r2_score(y, pred_y)
    return round(score, 2)*100

# Data generated
embeddings = []
losses = []
accuracies = []
outputs = []
targets = []

# Use GPU for training, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Data loader
data_size = len(dataset)
NUM_GRAPHS_PER_BATCH = 64
NUM_EPOCHS = 2000

torch.manual_seed(12345)

#randomize and split the data
dataset = dataset.shuffle()

train_dataset = dataset[:int(data_size * 0.8)]
test_dataset = dataset[int(data_size * 0.8):]

loader = DataLoader(train_dataset, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=False)

print('\n======== data distribution =======\n')
print("Size of training data: {} graphs".format(len(train_dataset)))
print("Size of testing data: {} graphs".format(len(test_dataset)))
 
def train(data):
    # Enumerate over the data
    for batch in loader:
      # Use GPU
      batch.to(device)  
      # Reset gradients
      optimizer.zero_grad() 
      # Passing the node features and the connection info
      pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 
      # Calculating the loss and gradients
      loss = loss_fn(pred, batch.y)     
      acc = r2_accuracy(pred.detach().numpy(), batch.y.detach().numpy())

      loss.backward()  
      # Update using the gradients
      optimizer.step()   
    return loss, acc, pred, batch.y, embedding

print('\n======== Starting training ... =======\n')
start_time = time.time()

losses = []
for epoch in range(NUM_EPOCHS):
    loss, acc, pred, target, h = train(data)
    losses.append(loss)
    accuracies.append(acc)
    outputs.append(pred)
    targets.append(target)
    
    if epoch % 100 == 0:
      # print(f"Epoch {epoch} | Train Loss {loss}")
      print(f'Epoch {epoch:>3} | Loss: {loss:.5f} | Acc: {acc:.2f}%')

print("\nTraining done!\n")
elapsed = time.time() - start_time
minutes_e = elapsed//60
print("--- training took:  %s minutes ---" % (minutes_e))
```

    
    ======== data distribution =======
    
    Size of training data: 3360 graphs
    Size of testing data: 840 graphs
    
    ======== Starting training ... =======
    
    Epoch   0 | Loss: 1.21428 | Acc: -1.00%
    Epoch 100 | Loss: 0.59290 | Acc: 60.00%
    Epoch 200 | Loss: 0.37887 | Acc: 64.00%
    Epoch 300 | Loss: 0.27566 | Acc: 82.00%
    Epoch 400 | Loss: 0.30305 | Acc: 80.00%
    Epoch 500 | Loss: 0.19684 | Acc: 87.00%
    Epoch 600 | Loss: 0.21954 | Acc: 88.00%
    Epoch 700 | Loss: 0.17783 | Acc: 86.00%
    Epoch 800 | Loss: 0.09046 | Acc: 92.00%
    Epoch 900 | Loss: 0.13527 | Acc: 91.00%
    Epoch 1000 | Loss: 0.11694 | Acc: 92.00%
    Epoch 1100 | Loss: 0.09441 | Acc: 92.00%
    Epoch 1200 | Loss: 0.08276 | Acc: 96.00%
    Epoch 1300 | Loss: 0.07728 | Acc: 96.00%
    Epoch 1400 | Loss: 0.05626 | Acc: 96.00%
    Epoch 1500 | Loss: 0.02418 | Acc: 99.00%
    Epoch 1600 | Loss: 0.03023 | Acc: 98.00%
    Epoch 1700 | Loss: 0.03533 | Acc: 97.00%
    Epoch 1800 | Loss: 0.02907 | Acc: 96.00%
    Epoch 1900 | Loss: 0.02873 | Acc: 97.00%
    
    Training done!
    
    --- training took:  48.0 minutes ---


<div style="text-align:left;color:Maroon">
    <h4>Create a dataframe for our training results for easy plotting. 
'outputs' and 'targets' are tensors and they need to be converted to arrays.</h4>
</div>


```python
losses_float = [float(loss.cpu().detach().numpy()) for loss in losses] 
losses_np = np.array([x.item() for x in losses])
outs = [i[0] for i in outputs]
outputs_np = np.array([x.item() for x in outs])
targs = [i[0] for i in targets]
targets_np = np.array([x.item() for x in targs])

results = pd.concat([pd.DataFrame(losses_np),
                     pd.DataFrame(accuracies),
                     pd.DataFrame(outputs_np),
                     pd.DataFrame(targets_np)], axis= 1)
results.columns = ['losses', 'accuracy', 'pred', 'target']
results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>losses</th>
      <th>accuracy</th>
      <th>pred</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.214280</td>
      <td>-1.0</td>
      <td>2.171925</td>
      <td>-1.08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.388352</td>
      <td>1.0</td>
      <td>2.217482</td>
      <td>3.15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.220752</td>
      <td>1.0</td>
      <td>2.012036</td>
      <td>3.14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.622855</td>
      <td>-0.0</td>
      <td>2.229038</td>
      <td>1.92</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.812663</td>
      <td>-7.0</td>
      <td>2.217885</td>
      <td>1.30</td>
    </tr>
  </tbody>
</table>
</div>



<div style="text-align:left;color:Maroon">
    <h4>y-axis limits so we can use the same scale for the whole data and the first 20 epochs (Zoomed in).</h4>
</div>


```python
ymin, ymax = np.floor(min(results[[ 'pred', 'target']].min())),\
    round(max(results[[ 'pred', 'target']].max()))
```

<div style="text-align:center;color:Blue">
    <h3> Evaluate model</h3>
</div>
 
#### The following granularization of scores is not always required but it helps to know how training is progressing


```python
# all training
training_acc = r2_accuracy(results["target"], results["pred"])

#first 20 epochs
training_acc_1st_20 = r2_accuracy(results["target"][:20], results["pred"][:20])

# last 20 epochs
training_acc_last_20 = r2_accuracy(results["target"][-20:], results["pred"][-20:])

print("Training accuracy: {}%".format(round(training_acc, 2)))
print("1st 20 Training accuracy: {}%".format(round(training_acc_1st_20, 2)))
print("Last 20 Training accuracy: {}%".format(round(training_acc_last_20, 2)))
```

    Training accuracy: 82.0%
    1st 20 Training accuracy: -8049.0%
    Last 20 Training accuracy: 98.0%


#### Note the -8049% training accuracy for the 20 first training epochs. Accuracy fluctuation is a common occurrence especially early in any model training. This could be for many reasons which can read on:
1. network issues : https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607
2. accuracy flactuation and over-fitting: https://medium.com/@dnyaneshwalwadkar/fix-training-accuracy-fluctuation-over-fitting-problem-in-deep-learning-algorithm-859573090809 

<div style="text-align:left;color:Maroon">
    <h4>Explore training results, visually.</h4>
</div>


```python
import matplotlib.pyplot as plt

# creating grid for subplots
fig = plt.figure(figsize=(10,6))
 
ax1 = plt.subplot2grid(shape=(2, 28), loc=(0,0), colspan=15)
ax2 = plt.subplot2grid(shape=(2, 28), loc=(0,17), colspan=4)
ax3 = plt.subplot2grid(shape=(2, 28), loc=(0,23), colspan=4)


results[[ 'pred', 'target']].plot(title='Training target and prediction values\n Training accuracy: ' + str(round(training_acc, 2) )+ '%',
                                  xlabel='epoch', ylabel = 'logD at pH 7.4', ax=ax1, ylim = (ymin, ymax) ) 
results[[ 'pred', 'target']][:20].plot(title="$1^{st}$ 20 epochs\nAccuracy: " + str(round(training_acc_1st_20, 2)) +'%',
                                       xlabel='epoch', ax=ax2, ylim = (ymin, ymax))
results[[ 'pred', 'target']][-20:].plot(title="Last 20 epochs\nAccuracy: " + str(round(training_acc_last_20, 2)) +'%',
                                       xlabel='epoch', ax=ax3, ylim = (ymin, ymax))
fig.savefig('assets/images/pred_vs_targ.png', bbox_inches='tight')
```


    
![png](assets/images/pred_vs_targ.png)
    


<div style="text-align:left;color:Maroon">
    <h4>More performance metrics.</h4>
</div>

<div style="text-align:left;color:Maroon">
    <h5>Plot RMSE (loss) and accuracy.</h5>
</div>
 


```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax1 = plt.subplots(figsize=(8,4))

color1 = 'tab:red'
ax1.set_xlabel('epochs')

ax1.tick_params(axis='y', labelcolor=color1)
sns.lineplot(data=results.accuracy, label="'Accuracy'", color=color1, ax=ax1)
plt.legend(loc='upper right')

ax2 = ax1.twinx() 

color2 = 'tab:blue'
ax1.set_ylabel('RMSE' , color=color2, labelpad=40)
ax2.set_ylabel("'Accuracy'", color=color1, labelpad=30)  

ax2.tick_params(axis='y', labelcolor=color2)
sns.lineplot(data=losses_float, label='Training loss', color=color2, ax=ax2)
plt.legend(loc='upper left')

ax1.yaxis.tick_right()
ax2.yaxis.tick_left()

plt.title("Training losses and 'accuracies' (approximated by 2 decimal closeness)")

plt.show()
fig.tight_layout()
fig.savefig('assets/images/losses_and_accuracies.png', bbox_inches='tight')
```


    
![png](assets/images/losses_and_accuracies.png)
    


<div style="text-align:center;color:Blue">
    <h3> Evaluate model: use test data</h3>
</div>

#### Now that the model is fully trained, a test data prediction can be performed to probe accuracy.
##### A few plots are shown below to help training and testing accuracy progression


```python
import pandas as pd 

# One batch prediction
test_batch = next(iter(test_loader))
with torch.no_grad():
    test_batch.to(device)
    pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch) 
    df = pd.DataFrame()
    df["y"] = test_batch.y.tolist()
    df["y_pred"] = pred.tolist()
df["real"] = df["y"].apply(lambda row: row[0])
df["pred"] = df["y_pred"].apply(lambda row: row[0])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>y_pred</th>
      <th>real</th>
      <th>pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[2.430000066757202]</td>
      <td>[2.0413427352905273]</td>
      <td>2.43</td>
      <td>2.041343</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[1.8300000429153442]</td>
      <td>[1.6838014125823975]</td>
      <td>1.83</td>
      <td>1.683801</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[1.909999966621399]</td>
      <td>[1.6316179037094116]</td>
      <td>1.91</td>
      <td>1.631618</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[2.1700000762939453]</td>
      <td>[2.0308873653411865]</td>
      <td>2.17</td>
      <td>2.030887</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0.9800000190734863]</td>
      <td>[3.1909775733947754]</td>
      <td>0.98</td>
      <td>3.190978</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_acc = r2_accuracy(df["real"], df["pred"])
test_acc_1st_20 = r2_accuracy(df["real"][:20], df["pred"][:20])

print("Test accuracy is {}%".format(round(test_acc, 2) ))
print("1st 20 test accuracy is {}%".format(round(test_acc_1st_20, 2)))
```

    Test accuracy is 49.0%
    1st 20 test accuracy is 14.0%



```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10,6))

ax1 = plt.subplot2grid(shape=(2, 36), loc=(0,0), colspan=15)
ax2 = plt.subplot2grid(shape=(2, 36), loc=(0,20), colspan=5)
ax3 = plt.subplot2grid(shape=(2, 36), loc=(0,30), colspan=5)

test_title = "Test target and prediction values." + "\nTest accuracy: {}%\n".format(round(test_acc, 2))
df[["real", "pred"]].plot(title=test_title, xlabel='sample', ylabel = 'logD at pH 7.4',
                          ax=ax1, ylim = (ymin, ymax) ) 

test_title_1st_20 = "$1^{st}$ 20 samples\n in testing." + "\nTest accuracy: {}%\n".format(round(test_acc_1st_20, 2))  
df[["real", "pred"]][:20].plot(title=test_title_1st_20, xlabel='sample',
                                       ax=ax2, ylim = (ymin, ymax))

train_title = "$1^{st}$ 20 epochs\n in training." + "\nTrain accuracy: {}%\n".format(round(training_acc_1st_20,2))  
results[[ 'pred', 'target']][:20].plot(title=train_title,
                                       xlabel='epoch',
                                 # ylabel = 'logD at pH 7.4',
                                       ax=ax3, ylim = (ymin, ymax))
fig.savefig('assets/images/pred_vs_targ_trained.png', bbox_inches='tight')
```


    
![png](assets/images/pred_vs_targ_trained.pngg)
    


### TO DO:
#### Note the low test accuracy value, which as mentioned before, could be due to over training (the last 20 samples during training were predicted at 98% accuracy). Try larger data for testing, smaller data and epochs for training


 ## Contributing and Permissions

Please feel free to reach out to me at https://www.linkedin.com/in/mulugeta-semework-abebe/ for ways to collaborate or use some components.

##

## Resources (thank you all for ideas and contents!)

Datasets: https://moleculenet.org/datasets-1

Python_geometric: https://github.com/pyg-team/pytorch_geometric/tree/master

https://towardsdatascience.com/a-beginners-guide-to-graph-neural-networks-using-pytorch-geometric-part-1-d98dc93e7742

https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial

https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing#scrollTo=9mTJL0WfzeBq

https://medium.com/@tejpal.abhyuday/application-of-gnn-for-calculating-the-solubility-of-molecule-graph-level-prediction-8bac5fabf600

https://medium.com/@tejpal.abhyuday/application-of-graph-neural-networks-for-node-classification-on-cora-dataset-f48bc09f1765

https://www.kaggle.com/code/salmaneunus/predict-binding-affinity-of-protein-ligand-systems

https://towardsdatascience.com/drug-discovery-with-graph-neural-networks-part-1-1011713185eb


