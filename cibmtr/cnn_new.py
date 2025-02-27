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
from cibmtr.preprocess import BooleanConverterTransformer, CMVSplitterTransformer, ConvertObjectToCategoricalTransformer, DataFrameMinMaxScaler, DropColumnsTransformer, DropHighCorrelationColumnsTransformer, DropSingleValueColumnsTransformer, FixBoolTypeTransformer, LabelEncoderTransformer, RemoveOutliersPercentilesTransformer, ReplaceTransformer, SexSplitterTransformer, TreatMissingValuesTransformer, mapTransformer
    
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

# def preprocess_data(train, test):
#     """
#     Create torch dataloaders to prepare data for training and evaluation.
#     """
#     X = train.drop(['efs', 'efs_time'], axis=1)  # Features
#     y = train[['efs', 'efs_time']]  # Target

#     preprocessing_pipeline.fit(X, y)
#     train_preprocessed = preprocessing_pipeline.transform(train)
#     # verificar gvhd_proph com um missing
#     test_preprocessed = preprocessing_pipeline.transform(test)
    
#     dl_train = init_dl(train_preprocessed, training=True)
#     dl_test = init_dl(test_preprocessed)
#     return dl_train, dl_test

def init_dl(df, training=False):
    """
    Initialize data loaders with 4 dimensions : categorical dataframe, numerical dataframe and target values (efs and efs_time).
    Notice that efs_time is log-transformed.
    Fix batch size to 2048 and return dataloader for training or validation depending on training value.
    """
    cols_cat, cols_num = get_feature_types(df)
    df_cat = df[cols_cat].copy()
    df_num = df[cols_num].copy().astype(np.float32)
    ds = TensorDataset(
        torch.tensor(df_cat.values, dtype=torch.long),
        torch.tensor(df_num.values, dtype=torch.float32),
        torch.tensor(df.efs_time.values, dtype=torch.float32).log(),
        torch.tensor(df.efs.values, dtype=torch.long)
    )
    bs = 2048
    # Teste bs = 512
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, pin_memory=True, shuffle=training)
    return dl

def get_feature_types(df):
    """
    Utility function to return categorical and numerical column names.
    """
    categorical_cols = [col for col in df.columns if (pd.api.types.is_categorical_dtype(df[col]))]
    # categorical_cols = [col for i, col in enumerate(train.columns) if ((train[col].dtype == "object") | (2 < train[col].nunique() < 25))]
    RMV = ["ID", "efs", "efs_time", "y"]
    FEATURES = [c for c in df.columns if not c in RMV]
    numerical_cols = [i for i in FEATURES if i not in categorical_cols]
    return categorical_cols, numerical_cols

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
    ('label_encoder', LabelEncoderTransformer()),
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

class CatEmbeddings(nn.Module):
    """
    Embedding module for the categorical dataframe.
    """
    def __init__(
        self,
        projection_dim: int,
        categorical_cardinality: List[int],
        max_embedding_dim: int
    ):
        """
        projection_dim: The dimension of the final output after projecting the concatenated embeddings.
        categorical_cardinality: A list of cardinalities for each categorical feature.
        embedding_dim: A LIST of embedding dimensions for each categorical feature, corresponding to categorical_cardinality.
        self.embeddings: list of embedding layers, now with different embedding dimensions.
        self.projection: Sequential network for projecting concatenated embeddings.
        """
        super(CatEmbeddings, self).__init__()
        # embedding_dim = [min(max_embedding_dim, 6 * int(pow(cardinality, 0.25))) for cardinality in categorical_cardinality]
        embedding_dim = [min(max_embedding_dim, cardinality) for cardinality in categorical_cardinality]
        self.embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(c, d),
                nn.LayerNorm(d)  # Normaliza os embeddings para evitar dominância
            )
            for c, d in zip(categorical_cardinality, embedding_dim) 
        ])
        # # Inicializa os embeddings corretamente
        # for embedding in self.embeddings:
        #     torch.nn.init.xavier_uniform_(embedding[0].weight)
        self.projection = nn.Sequential(
            nn.Linear(sum(embedding_dim), projection_dim), # Ajustado para usar a soma das embedding_dim como dimensão de entrada
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x_cat):
        """
        Apply projection on concatenated embeddings from all categorical features.
        """
        x_cat = [embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)]
        x_cat = torch.cat(x_cat, dim=1)
        return self.projection(x_cat)


class NN(nn.Module):
    """
    Train a model on both categorical embeddings and numerical data.
    """
    def __init__(
            self,
            continuous_dim: int,
            categorical_cardinality: List[int],
            max_embedding_dim: int,
            projection_dim: int,
            hidden_dim: int,
            dropout: float = 0
    ):
        """
        continuous_dim: The number of continuous features.
        categorical_cardinality: A list of integers representing the number of unique categories in each categorical feature.
        embedding_dim: The dimensionality of the embedding space for each categorical feature.
        projection_dim: The size of the projected output space for the categorical embeddings.
        hidden_dim: The number of neurons in the hidden layer of the MLP.
        dropout: The dropout rate applied in the network.
        self.embeddings: previous embeddings for categorical data.
        self.mlp: defines an MLP model with an ODST layer followed by batch normalization and dropout.
        self.out: linear output layer that maps the output of the MLP to a single value
        self.dropout: defines dropout
        Weights initialization with xavier normal algorithm and biases with zeros.
        """
        super(NN, self).__init__()
        self.embeddings = CatEmbeddings(projection_dim, categorical_cardinality, max_embedding_dim)
        self.mlp = nn.Sequential(
            ODST(projection_dim + continuous_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_cat, x_cont):
        """
        Create embedding layers for categorical data, concatenate with continous variables.
        Add dropout and goes through MLP and return raw output and 1-dimensional output as well.
        """
        x = self.embeddings(x_cat)
        x = torch.cat([x, x_cont], dim=1)
        x = self.dropout(x)
        x = self.mlp(x)
        return self.out(x), x


@functools.lru_cache
def combinations(N):
    """
    calculates all possible 2-combinations (pairs) of a tensor of indices from 0 to N-1, 
    and caches the result using functools.lru_cache for optimization
    """
    ind = torch.arange(N)
    comb = torch.combinations(ind, r=2)
    # return comb.cuda()
    return comb


class LitNN(pl.LightningModule):
    """
    Main Model creation and losses definition to fully train the model.
    """
    def __init__(
            self,
            continuous_dim: int,
            categorical_cardinality: List[int],
            max_embedding_dim: int,
            race_index: int,
            projection_dim: int,
            hidden_dim: int,
            lr: float = 1e-3,
            dropout: float = 0.2,
            weight_decay: float = 1e-3,
            aux_weight: float = 0.1,
            margin: float = 0.5,
    ):
        """
        continuous_dim: The number of continuous input features.
        categorical_cardinality: A list of integers, where each element corresponds to the number of unique categories for each categorical feature.
        projection_dim: The dimension of the projected space after embedding concatenation.
        hidden_dim: The size of the hidden layers in the feedforward network (MLP).
        lr: The learning rate for the optimizer.
        dropout: Dropout probability to avoid overfitting.
        weight_decay: The L2 regularization term for the optimizer.
        aux_weight: Weight used for auxiliary tasks.
        margin: Margin used in some loss functions.
        race_index: An index that refer to race_group in the input data.
        """
        super(LitNN, self).__init__()
        self.save_hyperparameters()
        
        # Creates an instance of the NN model defined above
        self.model = NN(
            continuous_dim = self.hparams.continuous_dim,
            categorical_cardinality = self.hparams.categorical_cardinality,
            max_embedding_dim = self.hparams.max_embedding_dim,
            projection_dim=self.hparams.projection_dim,
            hidden_dim=self.hparams.hidden_dim,
            dropout=self.hparams.dropout
        )
        self.targets = []

        # Defines a small feedforward neural network that performs an auxiliary task with 1-dimensional output
        self.aux_cls = nn.Sequential(
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim // 3),
            nn.GELU(),
            nn.Linear(self.hparams.hidden_dim // 3, 1)
        )

    def on_before_optimizer_step(self, optimizer):
        """
        Compute the 2-norm for each layer
        If using mixed precision, the gradients are already unscaled here
        """
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def forward(self, x_cat, x_cont):
        """
        Forward pass that outputs the 1-dimensional prediction and the embeddings (raw output)
        """
        x, emb = self.model(x_cat, x_cont)
        return x.squeeze(1), emb

    def training_step(self, batch, batch_idx):
        """
        defines how the model processes each batch of data during training.
        A batch is a combination of : categorical data, continuous data, efs_time (y) and efs event.
        y_hat is the efs_time prediction on all data and aux_pred is auxiliary prediction on embeddings.
        Calculates loss and race_group loss on full data.
        Auxiliary loss is calculated with an event mask, ignoring efs=0 predictions and taking the average.
        Returns loss and aux_loss multiplied by weight defined above.
        """
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        aux_pred = self.aux_cls(emb).squeeze(1)
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        aux_loss = nn.functional.mse_loss(aux_pred, y, reduction='none') # TODO função de custo do score considerando a raça
        aux_mask = efs == 1
        aux_loss = (aux_loss * aux_mask).sum() / aux_mask.sum()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("race_loss", race_loss, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        self.log("aux_loss", aux_loss, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        return loss + aux_loss * self.hparams.aux_weight

    def get_full_loss(self, efs, x_cat, y, y_hat):
        """
        Output loss and race_group loss.
        """
        loss = self.calc_loss(y, y_hat, efs)
        race_loss = self.get_race_losses(efs, x_cat, y, y_hat)
        loss += 0.1 * race_loss
        return loss, race_loss

    def get_race_losses(self, efs, x_cat, y, y_hat):
        """
        Calculate loss for each race_group based on deviation/variance.
        """
        races = torch.unique(x_cat[:, self.hparams.race_index])
        race_losses = []
        for race in races:
            ind = x_cat[:, self.hparams.race_index] == race
            race_losses.append(self.calc_loss(y[ind], y_hat[ind], efs[ind]))
        race_loss = sum(race_losses) / len(race_losses)
        races_loss_std = sum((r - race_loss)**2 for r in race_losses) / len(race_losses)
        return torch.sqrt(races_loss_std)

    def calc_loss(self, y, y_hat, efs):
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
        N = y.shape[0]
        comb = combinations(N)
        comb = comb[(efs[comb[:, 0]] == 1) | (efs[comb[:, 1]] == 1)]
        pred_left = y_hat[comb[:, 0]]
        pred_right = y_hat[comb[:, 1]]
        y_left = y[comb[:, 0]]
        y_right = y[comb[:, 1]]
        y = 2 * (y_left > y_right).int() - 1
        loss = nn.functional.relu(-y * (pred_left - pred_right) + self.hparams.margin)
        mask = self.get_mask(comb, efs, y_left, y_right)
        loss = (loss.double() * (mask.double())).sum() / mask.sum()
        return loss

    def get_mask(self, comb, efs, y_left, y_right):
        """
        Defines all invalid comparisons :
        * Case 1: "Left outlived Right" but Right is censored
        * Case 2: "Right outlived Left" but Left is censored
        Masks for case 1 and case 2 are combined using |= operator and inverted using ~ to create a "valid pair mask"
        """
        left_outlived = y_left >= y_right
        left_1_right_0 = (efs[comb[:, 0]] == 1) & (efs[comb[:, 1]] == 0)
        mask2 = (left_outlived & left_1_right_0)
        right_outlived = y_right >= y_left
        right_1_left_0 = (efs[comb[:, 1]] == 1) & (efs[comb[:, 0]] == 0)
        mask2 |= (right_outlived & right_1_left_0)
        mask2 = ~mask2
        mask = mask2
        return mask

    def validation_step(self, batch, batch_idx):
        """
        This method defines how the model processes each batch during validation
        """
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        self.targets.append([y, y_hat.detach(), efs, x_cat[:, self.hparams.race_index]])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        """
        At the end of the validation epoch, it computes and logs the concordance index
        """
        cindex, metric = self._calc_cindex()
        self.log("cindex", metric, on_epoch=True, prog_bar=True, logger=True)
        self.log("cindex_simple", cindex, on_epoch=True, prog_bar=True, logger=True)
        self.targets.clear()

    def _calc_cindex(self):
        """
        Calculate c-index accounting for each race_group or global.
        """
        y = torch.cat([t[0] for t in self.targets]).cpu().numpy()
        y_hat = torch.cat([t[1] for t in self.targets]).cpu().numpy()
        efs = torch.cat([t[2] for t in self.targets]).cpu().numpy()
        races = torch.cat([t[3] for t in self.targets]).cpu().numpy()
        metric = self._metric(efs, races, y, y_hat)
        cindex = concordance_index(y, y_hat, efs)
        return cindex, metric

    def _metric(self, efs, races, y, y_hat):
        """
        Calculate c-index accounting for each race_group
        """
        metric_list = []
        for race in np.unique(races):
            y_ = y[races == race]
            y_hat_ = y_hat[races == race]
            efs_ = efs[races == race]
            metric_list.append(concordance_index(y_, y_hat_, efs_))
        metric = float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))
        return metric

    def test_step(self, batch, batch_idx):
        """
        Same as training step but to log test data
        """
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        self.targets.append([y, y_hat.detach(), efs, x_cat[:, self.hparams.race_index]])
        self.log("test_loss", loss)
        return loss

    def on_test_epoch_end(self) -> None:
        """
        At the end of the test epoch, calculates and logs the concordance index for the test set
        """
        cindex, metric = self._calc_cindex()
        self.log("test_cindex", metric, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_cindex_simple", cindex, on_epoch=True, prog_bar=True, logger=True)
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

def generate_model(dl_train, dl_val, continuous_dim, categorical_cardinality, race_index, hparams=None):
    """
    Defines model hyperparameters and fit the model.
    """
    if hparams is None:
        hparams = {
            "max_embedding_dim": 16,
            # "projection_dim": 112,
            "projection_dim": 8,
            "hidden_dim": 56,
            "lr": 0.06464861983337984,
            "dropout": 0.05463240181423116,
            "aux_weight": 0.26545778308743806,
            "margin": 0.2588153271003354,
            "weight_decay": 0.0002773544957610778
        }
    model = LitNN(
        continuous_dim = continuous_dim,
        categorical_cardinality = categorical_cardinality,
        race_index = race_index,
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
    trainer.fit(model, dl_train) # TODO Testar com, val_dataloaders=dl_val)
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
    train_original = general_preprocessing_pipeline.fit_transform(train)
    test = general_preprocessing_pipeline.transform(test)
    cols_cat, cols_num = get_feature_types(train_original)
    continuous_dim = len(cols_num)
    categorical_cardinality = [len(train_original[col].cat.categories) for col in cols_cat]
    race_index = cols_cat.index(race_col_name)
    test_pred = np.zeros(test.shape[0])
    kf = StratifiedKFold(n_splits=5, shuffle=True, )
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
        
        dl_train = init_dl(train_preprocessed, training=True)
        dl_val = init_dl(val_preprocessed)
        model = generate_model(dl_train, dl_val, continuous_dim, categorical_cardinality, race_index, hparams)
        # Create submission
        # pred, _ = model.cuda().eval()(
        #     torch.tensor(X_cat_val, dtype=torch.long).cuda(),
        #     torch.tensor(X_num_val, dtype=torch.float32).cuda()
        # )
        test_preprocessed = preprocessing_pipeline.transform(test)
        dl_test = init_dl(test_preprocessed)
        pred, _ = model.eval()(dl_test)
        test_pred += pred.detach().cpu().numpy()
        
    subm_data = pd.read_csv("../input/equity-post-HCT-survival-predictions/sample_submission.csv")
    subm_data['prediction'] = -test_pred
    subm_data.to_csv('submission.csv', index=False)
    
    # display(subm_data.head())
    return 

hparams = None
hparams = {
    "max_embedding_dim": 16,
    # "projection_dim": 112,
    "projection_dim": 8,
    "hidden_dim": 56,
    "lr": 0.06464861983337984,
    "dropout": 0.05463240181423116,
    "aux_weight": 0.26545778308743806,
    "margin": 0.2588153271003354,
    "weight_decay": 0.0002773544957610778
}
max_embedding_dims = range(2, 18, 2)
projection_dims = range(2, 76, 2)
hidden_dims = range(2, 100, 2)
for max_embedding_dim in max_embedding_dims:
    for projection_dim in projection_dims:
        for hidden_dim in hidden_dims:
            hparams = {
                "max_embedding_dim": max_embedding_dim,
                "projection_dim": projection_dim,
                "hidden_dim": hidden_dim,
                "lr": 0.06464861983337984,
                "dropout": 0.05463240181423116,
                "aux_weight": 0.26545778308743806,
                "margin": 0.2588153271003354,
                "weight_decay": 0.0002773544957610778
            }
            print(hparams)
            try:
                main(hparams)
            except:
                print("Error")
            print("done")
# res = main(hparams)
print("done")