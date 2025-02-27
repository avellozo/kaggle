import os
import sys
import numpy as np
import pandas as pd

import functools
from typing import List

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold

from warnings import filterwarnings
sys.path.append(os.path.abspath(os.path.join('.')))
from cibmtr.preprocess import BooleanConverterTransformer, CMVSplitterTransformer, ConvertObjectToCategoricalTransformer, DataFrameMinMaxScaler, DropColumnsTransformer, DropHighCorrelationColumnsTransformer, DropSingleValueColumnsTransformer, FixBoolTypeTransformer, LabelEncoderTransformer, OneHotEncodeCategoricalTransformer, RemoveOutliersPercentilesTransformer, ReplaceTransformer, SexSplitterTransformer, TreatMissingValuesTransformer, mapTransformer
from cibmtr.tools import calc_score

import torch
from torch.utils.data import TensorDataset
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.cli import ReduceLROnPlateau
from pytorch_tabular.models.common.layers import ODST
from pytorch_lightning.utilities import grad_norm

from lifelines.utils import concordance_index

race_col_name = 'race_group'
data_path = 'cibmtr/'

filterwarnings('ignore')

def init_dl(df, training=False):
    """
    Initialize data loaders with 4 dimensions : numerical DATA , target values (efs and efs_time) and df['race_group'].
    Notice that efs_time is log-transformed.
    Fix batch size to 2048 and return dataloader for training or validation depending on training value.
    """
    # df = pd.get_dummies(df, columns=df.select_dtypes(include=['category']).columns)
    df_num = df.drop(columns=['efs', 'efs_time', 'race_group']).copy().astype(np.float32)
    ds = TensorDataset(
        torch.tensor(df_num.values, dtype=torch.float32),
        torch.tensor(df.efs_time.values, dtype=torch.float32).log1p(),
        torch.tensor(df.efs.values, dtype=torch.long),
        torch.tensor(df.race_group.values, dtype=torch.long),
    )
    bs = 2048
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, pin_memory=True, shuffle=training)
    return dl

def add_features(df):
    """
    Create some new features to help the model focus on specific patterns.
    """
    # Testar nova features
    
    # sex_match = df.sex_match.astype(str)
    # sex_match = sex_match.str.split("-").str[0] == sex_match.str.split("-").str[1]
    # df['sex_match_bool'] = sex_match
    # df.loc[df.sex_match.isna(), 'sex_match_bool'] = np.nan
    # df['big_age'] = df.age_at_hct > 16
    # df.loc[df.year_hct == 2019, 'year_hct'] = 2020
    df['is_cyto_score_same'] = (df['cyto_score'] == df['cyto_score_detail']).astype(int)
    # df['strange_age'] = df.age_at_hct == 0.044
    # df['age_bin'] = pd.cut(df.age_at_hct, [0, 0.0441, 16, 30, 50, 100])
    # df['age_ts'] = df.age_at_hct / df.donor_age
    df['year_hct'] -= 2000
    
    return df

mappings = {
    'dri_score': {'Low': 0, 'Intermediate': 1, 'Intermediate - TED AML case <missing cytogenetics': 1,
                  'High': 2, 'High - TED AML case <missing cytogenetics': 2, 'Very high': 3, 
                  'TBD': pd.NA, 'N/A - non-malignant indication': pd.NA, 'N/A - pediatric':pd.NA, 'TBD cytogenetics': pd.NA,
                  'N/A - disease not classifiable': pd.NA, 'Missing disease status': pd.NA},
    'cyto_score':{'Poor': 0, 'Intermediate':1, 'Normal': 1, 'Favorable':2, 'TBD': pd.NA, 'Other': pd.NA, 'Not tested': pd.NA},
    'cyto_score_detail': {'Poor': 0, 'Intermediate':1, 'Normal': 1, 'Favorable':2, 'TBD': pd.NA, 'Other': pd.NA, 'Not tested': pd.NA},
    'ethnicity': {'Not Hispanic or Latino':1, 'Hispanic or Latino':0, 'Non-resident of the U.S.':0},
    'sex_match': {'M-M': True, 'F-F': True, 'F-M': False, 'M-F':False},
    'cmv_status': {'+/+': True, '-/-': True, '+/-': False, '-/+':False},
    'donor_related': {'Unrelated': False, 'Related': True, 'Multiple donor (non-UCB)': False, 'Multiple donor (UCB)': False},
    'melphalan_dose': {'N/A, Mel not given': False, 'MEL': True},
    
}

columns_to_drop = ['ID',
    'hla_high_res_10', 'hla_high_res_8', 'hla_high_res_6', 'hla_low_res_6', 'hla_low_res_8', 'hla_low_res_10', 'hla_nmdp_6']

general_preprocessing_pipeline = Pipeline(steps=[
    ('drop_columns_ID', DropColumnsTransformer(columns_to_drop=columns_to_drop)),
    ('replace_not_done', ReplaceTransformer(value_to_replace='Not done', replacement=np.nan)),
    ('sex_splitter', SexSplitterTransformer()),
    ('cmv_splitter', CMVSplitterTransformer()),
    ('fix_bool', FixBoolTypeTransformer()),
    ('convert_mappings', mapTransformer(mappings)),
    ('convert_to_categorical', ConvertObjectToCategoricalTransformer(columns_to_exclude=[])),
    ('convert_cat_to_bool', BooleanConverterTransformer()),
    # ('label_encoder', LabelEncoderTransformer()),
    ('ohe_converter', OneHotEncodeCategoricalTransformer())
])

preprocessing_pipeline = Pipeline(steps=[
    ('drop_single_value', DropSingleValueColumnsTransformer()),
    ('remove_outliers', RemoveOutliersPercentilesTransformer(percentile=0.20)),
    ('treat_missing', TreatMissingValuesTransformer()),
    ('drop_high_corr', DropHighCorrelationColumnsTransformer(threshold=0.80, undrop_cols=[])),
    ('scaler', DataFrameMinMaxScaler()),
])

def load_data():
    """
    Load data and add features.
    """
    test = pd.read_csv(f"{data_path}input/equity-post-HCT-survival-predictions/test.csv")
    test = add_features(test)
    train = pd.read_csv(f"{data_path}input/equity-post-HCT-survival-predictions/train.csv")
    train = add_features(train)
    
    print("Test shape:", test.shape)
    print("Train shape:", train.shape)
    return test, train

class NN(nn.Module):
    """
    Train a model on both categorical embeddings and numerical data.
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            dropout: float = 0
    ):
        """
        input_dim: The number of input features.
        hidden_dim: The number of neurons in the hidden layer of the MLP.
        dropout: The dropout rate applied in the network.
        """
        super(NN, self).__init__()
        self.mlp = nn.Sequential(
            ODST(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass through the MLP.
        """
        x = self.dropout(x)
        x = self.mlp(x)
        return self.out(x)

class LitNN(pl.LightningModule):
    """
    Main Model creation and losses definition to fully train the model.
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            lr: float = 1e-3,
            dropout: float = 0.2,
            weight_decay: float = 1e-3,
            # margin: float = 0.5,
    ):
        """
        input_dim: The number of input features.
        hidden_dim: The size of the hidden layers in the feedforward network (MLP).
        lr: The learning rate for the optimizer.
        dropout: Dropout probability to avoid overfitting.
        weight_decay: The L2 regularization term for the optimizer.
        margin: Margin used in some loss functions.
        """
        super(LitNN, self).__init__()
        self.save_hyperparameters()
        
        # Creates an instance of the NN model defined above
        self.model = NN(
            input_dim = self.hparams.input_dim,
            hidden_dim = self.hparams.hidden_dim,
            dropout = self.hparams.dropout
        )
        self.targets = []

    def on_before_optimizer_step(self, optimizer):
        """
        Compute the 2-norm for each layer
        If using mixed precision, the gradients are already unscaled here
        """
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def forward(self, x):
        """
        Forward pass that outputs the 1-dimensional prediction.
        """
        return self.model(x).squeeze(1)

    def training_step(self, batch, batch_idx):
        """
        defines how the model processes each batch of data during training.
        A batch is a combination of : continuous data, efs_time (y) and efs event.
        y_hat is the efs_time prediction on all data.
        Calculates loss and race_group loss on full data.
        Auxiliary loss is calculated with an event mask, ignoring efs=0 predictions and taking the average.
        Returns loss and aux_loss multiplied by weight defined above.
        """
        x, y, efs, race_series = batch
        y_hat = self(x)
        loss = self.get_loss(efs, race_series, y, y_hat)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def get_loss(self, efs, race_group_series, y, y_hat):
        """
        Calculate loss for each race_group based on deviation/variance.
        """
        races = torch.unique(race_group_series)
        scores = []
        for race in races:
            ind = (race_group_series == race).nonzero(as_tuple=True)[0]  # Indexação correta
            scores.append(self.calcule_score(y[ind], y_hat[ind], efs[ind]))
        # Converte scores para tensor e força requires_grad=True se necessário
        scores_tensor = torch.tensor(scores, dtype=torch.float32, device=y_hat.device)
        if not scores_tensor.requires_grad:
            scores_tensor.requires_grad_(True)
            # Calcula a loss garantindo que tudo está no grafo computacional
        score_total = 1 - (torch.mean(scores_tensor) - torch.sqrt(torch.var(scores_tensor)))

        return score_total 

    def calcule_score(self, y, y_hat, efs):
        """
        Most important part of the model : loss function used for training.
        We face survival data with event indicators along with time-to-event.

        This function computes the main loss by the following the steps :
        * create all data pairs with "combinations" function (= all "two subjects" combinations)
        * make sure that we have at least 1 event in each pair
        * convert y to +1 or -1 depending on the correct ranking
        * loss is computed using a margin-based hinge loss
        * mask is applied to ensure only valid pairs are being used (censored data can't be ranked with event in some cases)
        * average loss on all pairs is returned
        """

        return concordance_index(y.cpu().detach().numpy(), -y_hat.cpu().detach().numpy(), efs.cpu().detach().numpy())

    def validation_step(self, batch, batch_idx):
        """
        This method defines how the model processes each batch during validation
        """
        x, y, efs, race_series = batch
        y_hat = self(x)
        loss = self.get_loss(efs, race_series, y, y_hat)
        self.targets.append([y, y_hat.detach(), efs, race_series])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        """
        At the end of the validation epoch, it computes and logs the concordance index and calc_score
        """
        y = torch.cat([t[0] for t in self.targets])
        y_hat = torch.cat([t[1] for t in self.targets])
        efs = torch.cat([t[2] for t in self.targets])
        race_series = torch.cat([t[3] for t in self.targets])
        val_score = self.get_loss(self, efs, race_series, y, y_hat)
        self.log("val_score", val_score, on_epoch=True, prog_bar=True, logger=True)
        
        self.targets.clear()


    def test_step(self, batch, batch_idx):
        """
        Same as training step but to log test data
        """
        x, y, efs, race_series = batch
        y_hat = self(x)
        loss = self.get_loss(efs, race_series, y, y_hat)
        self.targets.append([y, y_hat.detach(), efs, race_series])
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_test_epoch_end(self) -> None:
        """
        At the end of the test epoch, calculates and logs the concordance index for the test set
        """
        y = torch.cat([t[0] for t in self.targets])
        y_hat = torch.cat([t[1] for t in self.targets])
        efs = torch.cat([t[2] for t in self.targets])
        race_series = torch.cat([t[3] for t in self.targets])
        val_score = self.get_loss(efs, race_series, y, y_hat)
        self.log("test_val_loss", val_score, on_epoch=True, prog_bar=True, logger=True)
        
        self.targets.clear()


    def configure_optimizers(self):
        """
        configures the optimizer and learning rate scheduler:
        * Optimizer: Adam optimizer with weight decay (L2 regularization).
        * Scheduler: Cosine Annealing scheduler, which adjusts the learning rate according to a cosine curve.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=45,
                eta_min=6e-3
            ),
            "interval": "epoch",
            "frequency": 1,
            "strict": False,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

def generate_model(dl_train, dl_val, input_dim, hparams=None):
    """
    Defines model hyperparameters and fit the model.
    """
    if hparams is None:
        hparams = {
            "hidden_dim": 56,
            "lr": 0.06464861983337984,
            "dropout": 0.05463240181423116,
            # "aux_weight": 0.26545778308743806,
            # "margin": 0.2588153271003354,
            "weight_decay": 0.0002773544957610778
        }
    model = LitNN(
        input_dim = input_dim,
        **hparams
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1)
    trainer = pl.Trainer(
        # accelerator='cuda',
        accelerator='cpu',
        max_epochs=55,
        log_every_n_steps=6,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval='epoch'),
            TQDMProgressBar(),
            StochasticWeightAveraging(swa_lrs=1e-5, swa_epoch_start=40, annealing_epochs=15)
        ],
    )
    trainer.fit(model, dl_train)
    trainer.test(model, dl_val)
    return model.eval()

pl.seed_everything(42)

def main(hparams):
    """
    Main function to train the model.
    The steps are as following :
    * load data and fill efs and efs time for test data with 1
    * initialize pred array with 0
    * get categorical and numerical columns
    * split the train data on the stratified criterion : race_group * newborns yes/no
    * preprocess the fold data (create dataloaders)
    * train the model and create final submission output
    """
    test, train = load_data()
    test['efs_time'] = 1
    test['efs'] = 1
    le_race = LabelEncoder()
    race_group_train = le_race.fit_transform(train["race_group"].astype(str))
    race_group_test= le_race.transform(test["race_group"].astype(str))
    train_original = general_preprocessing_pipeline.fit_transform(train)
    test = general_preprocessing_pipeline.transform(test)
    train_original["race_group"] = race_group_train
    test["race_group"] = race_group_test
    test_pred = np.zeros(test.shape[0])
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    for i, (train_index, test_index) in enumerate(
        kf.split(
            train_original, train_original.race_group.astype(str) + (train_original.age_at_hct == 0.044).astype(str)
        )
        ):
        tt = train_original.copy()
        train = tt.iloc[train_index]
        val = tt.iloc[test_index]
        
        
        train_preprocessed = preprocessing_pipeline.fit_transform(train)
        val_preprocessed = preprocessing_pipeline.transform(val)
        input_dim = train_preprocessed.shape[1] - 3  # Excluding 'efs' ,'efs_time' and 'race_group'
        
        dl_train = init_dl(train_preprocessed, training=True)
        dl_val = init_dl(val_preprocessed)
        model = generate_model(dl_train, dl_val, input_dim, hparams)
        # Create submission
        # pred, _ = model.cuda().eval()(
        #     torch.tensor(X_cat_val, dtype=torch.long).cuda(),
        #     torch.tensor(X_num_val, dtype=torch.float32).cuda()
        # )
        test_preprocessed = preprocessing_pipeline.transform(test)
        dl_test = init_dl(test_preprocessed)
        # Lista para armazenar todos os tensores df_num
        all_df_num = []

        # Iterar pelo DataLoader e extrair df_num
        for batch in dl_test:
            df_num_batch, _, _, _ = batch  # Desempacota o lote
            all_df_num.append(df_num_batch)

        # Concatenar todos os tensores em um único tensor
        df_num_tensor = torch.cat(all_df_num, dim=0)
        pred = model.eval()(df_num_tensor)
        test_pred += pred.detach().cpu().numpy()
        
    subm_data = pd.read_csv(f"{data_path}input/equity-post-HCT-survival-predictions/sample_submission.csv")
    subm_data['prediction'] = -test_pred
    subm_data.to_csv('submission.csv', index=False)
    
    return 

hparams = None
# hparams = {
#     "max_embedding_dim": 16,
#     # "projection_dim": 112,
#     "projection_dim": 8,
#     "hidden_dim": 56,
#     "lr": 0.06464861983337984,
#     "dropout": 0.05463240181423116,
#     # "aux_weight": 0.26545778308743806,
#     # "margin": 0.2588153271003354,
#     "weight_decay": 0.0002773544957610778
# }
# max_embedding_dims = range(2, 18, 2)
# projection_dims = range(2, 76, 2)
# hidden_dims = range(2, 100, 2)
# for max_embedding_dim in max_embedding_dims:
#     for projection_dim in projection_dims:
#         for hidden_dim in hidden_dims:
#             hparams = {
#                 "max_embedding_dim": max_embedding_dim,
#                 "projection_dim": projection_dim,
#                 "hidden_dim": hidden_dim,
#                 "lr": 0.06464861983337984,
#                 "dropout": 0.05463240181423116,
#                 # "aux_weight": 0.26545778308743806,
#                 # "margin": 0.2588153271003354,
#                 "weight_decay": 0.0002773544957610778
#             }
#             print(hparams)
#             try:
#                 main(hparams)
#             except:
#                 print("Error")
#             print("done")
res = main(hparams)
print("done")