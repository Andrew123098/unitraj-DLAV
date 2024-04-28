# Milestone 1

**Writen By: Andrew Brown and Mònica Laplana
  Class: Deep Learning for Autonomous Vehicles
  Code: CIVIL-459
  Professor: Dr. Alexandre Alahi**

## Table of Contents
- [Summary](#summary)
- [Results](#results)
- [Important Code We Added](#important-code-we-added)
- [How to Run the Code](#how-to-run-the-code)
- [Submission](#submission)

## Summary
In this code, we have successfully implemented the temporal and social attention functions in the transformer-based
trajectory prediction deep learning model. 

## Results
Best minADE6: ~1.5

## Important Code We Added
In the _forward() function we have implemented the calling of both of the attention functions.
We also implemented the Temporal and Social Attention functions from scratch. You can see the implmetations below.

### Put through Temporal and Attention Layers
```bash
for layer in range(self.L_enc):
    agents_emb = self.temporal_attn_fn(agents_emb, opps_masks, self.temporal_attn_layers[layer])
    agents_emb = self.social_attn_fn(agents_emb, opps_masks, self.social_attn_layers[layer])
```

### Definition of Temporal Attention Function
```bash
def temporal_attn_fn(self, agents_emb, agent_masks, layer):
    '''
    Gets agents embeddings and agents mask, and applies the temporal attention layer per agent.
    Make sure to apply the agent mask in the layer function (you could use src_key_padding_mask argument).
    Also don't forget to use positional encoding.
    :param agents_emb: (T, B, N, H) # The dimensions are Time (T), Batch size (B), Number of agents (N), and Embedding size (H).
    :param agent_masks: (B, T, N)
    :return: (T, B, N, H)
    '''
    ######################## Your code here ########################
    # The dimensions are Time (T), Batch size (B), Number of agents (N), and Embedding size (H).
    T, B, N, H = agents_emb.size()

    # Put into (T, N*B, H) shape
    agents_emb = agents_emb.reshape(T, N * B, H)
    
    # Pass to positional encoder
    agents_emb = self.pos_encoder.forward(agents_emb)

    # To avoid NaN from softmax function, set first timesteps to False (B, T, N)
    agent_masks[:,0,:] = False

    # Flatten the masks
    agent_masks = agent_masks.permute(1, 0, 2).reshape(B*N, T)

    # Apply temporal attention layer
    agents_emb = layer(agents_emb, src_key_padding_mask=agent_masks)

    # Reshape the embeddings back to their original shape
    agents_emb = agents_emb.view(T, B, N, H)
    
    ################################################################
    return agents_emb

```

### Definition of Social Attention Function:
```bash
def social_attn_fn(self, agents_emb, agent_masks, layer):
    '''
    Gets agents embeddings and agents mask, and applies the social attention layer per time step.
    Make sure to apply the agent mask in the layer function (you could use src_key_padding_mask argument).
    You don't need to use positional encoding here.
    :param agents_emb: (T, B, N, H)
    :param agent_masks: (B, T, N)
    :return: (T, B, N, H)
    '''
    ######################## Your code here ########################
    # (NOTE: We do not need to include the agent masks here because the masking is already applied in the temporal attn fn)
    T, B, N, H = agents_emb.size()

    # Put agents into size (N, B*T, H)
    agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(N, B*T, H) # N, B*T, H

    # Apply social attention layer 
    agents_emb = layer(agents_emb)

    # Reshape the embeddings back to their original shape
    agents_emb = agents_emb.view(N, B, T, -1).permute(2, 1, 0, 3) # T, B, N, H
    ################################################################
    return agents_emb
```


## How to Run the Code

### Installation

First start by cloning the repository:
```bash
git clone https://github.com/Andrew123098/unitraj-DLAV.git
cd unitraj-DLAV
```

Then make a virtual environment and install the required packages. 
```bash
python3 -m venv venv
source venv/bin/activate

# Install MetaDrive Simulator
cd ~/  # Go to the folder you want to host these two repos.
git clone https://github.com/metadriverse/metadrive.git
cd metadrive
pip install -e .

# Install ScenarioNet
cd ~/  # Go to the folder you want to host these two repos.
git clone https://github.com/metadriverse/scenarionet.git
cd scenarionet
pip install -e .
```

Finally, install Unitraj and login to wandb via:
```bash
cd unitraj-DLAV # Go to the folder you cloned the repo
pip install -r requirements.txt
pip install -e .
wandb login
```
If you don't have a wandb account, you can create one [here](https://wandb.ai/site). It is a free service for open-source projects and you can use it to log your experiments and compare different models easily.


You can verify the installation of UniTraj via running the training script:
```bash
python train.py method=ptr
```
The incomplete PTR model will be trained on several samples of data available in `motionnet/data_samples`. The data can also be accessed [here](https://drive.google.com/file/d/1mBpTqM5e_Ct6KWQenPUvNUBJWHn3-KUX/view?usp=sharing) .

### Running the Code
1. Change Model in motionnnet-->models-->ptr-->ptr.py
2. Change Hyperparemeters in motionnet-->configs-->method-->ptr.yaml
3. Change Run Configuration in motionnet-->configs-->config.yaml
4. Run one of the following commands:

#### To Train a Model:
If training an existing model change `ckpt_path` variable in `config.yaml`. otherwise leave as null
```bash
python train.py method=ptr
```
#### To Evaluate a Model:
```bash
python evaluation.py method=ptr
```

#### To Generate Some Basic Visualizations:
```bash
python tsne_visualization.py method=ptr
```

#### To Plot minADE6 Over Epochs:
```bash
python generate_loss_plots.py
```
#### To Generate a Submission File:
```bash
python gnerate_predictions.py method=ptr
```

## Submission

You can follow the steps in the [easy kaggle competition](https://www.kaggle.com/competitions/dlav-vehicle-trajectory-prediction-2024/overview) to submit the results and compare them with the other students in the leaderboard.

To generate the submission file run the following command:
```bash
python generate_predictions.py method=ptr
```
Before running the above command however, you need to put the path to the checkpoint of your trained model on the config file under `ckpt_path`. You can find the checkpoint of your trained model in the `lightning_logs` directory in the root directory of the project. 
For example, if you have trained your model for 10 epochs, you will find the checkpoint in `lightning_logs/version_0/checkpoints/epoch=10-val/brier_fde=30.93.ckpt`. You need to put the path to this file in the `config.py` file.

Additionally, for the `val_data_path` in the `config.yaml` file, you need to put the path to the test data you want to evaluate your model on.

The script will generate a file called `submission.csv` in the root directory of the project. You can submit this file to the kaggle competition. As this file could be big, we suggest you to compress it before submitting it.
