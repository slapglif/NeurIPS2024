from typing import Optional, Tuple, Dict, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen
from rdkit.Chem.Descriptors import ExactMolWt, HeavyAtomMolWt, MolWt, NumValenceElectrons, NumRadicalElectrons, \
    MinAbsPartialCharge, MaxAbsPartialCharge
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect, CalcNumRings, CalcNumAmideBonds, \
    CalcNumSpiroAtoms
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

torch.set_float32_matmul_precision('medium')


class StreamingMoleculeDataset(torch.utils.data.IterableDataset):
    """
    Dataset for streaming and processing molecule data from CSV or Parquet files.
    Includes calculation of LogP and selective Tanimoto similarity.
    """

    def __init__(self, file_path: str, smiles_col: str = 'molecule_smiles', target_col: Optional[str] = 'binds',
                 protein_col: str = 'protein_name', chunk_size: int = 10000, max_ref_mols: int = 1000):
        """
        Initialize the StreamingMoleculeDataset.

        Args:
            file_path (str): Path to the data file (CSV or Parquet).
            smiles_col (str): Name of the column containing SMILES strings.
            target_col (str): Name of the column containing target values.
            protein_col (str): Name of the column containing protein names.
            chunk_size (int): Number of rows to read per chunk.
            max_ref_mols (int): Maximum number of reference molecules for Tanimoto similarity.
        """
        super().__init__()
        self.file_path = file_path
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.protein_col = protein_col
        self.chunk_size = chunk_size
        self.max_ref_mols = max_ref_mols
        self.protein_encoder = LabelEncoder()
        self.reference_fps: Dict[str, DataStructs.ExplicitBitVect] = {}
        self._preprocess()

    def _preprocess(self) -> None:
        """
        Preprocess the dataset: fit the protein encoder and calculate reference fingerprints.
        """
        if self.file_path.endswith('.csv'):
            df = pd.read_csv(self.file_path, nrows=1000000)  # Load a subset for preprocessing
        elif self.file_path.endswith('.parquet'):
            columns = [self.protein_col, self.smiles_col]
            if self.target_col is not None:
                columns.append(self.target_col)
            df = pd.read_parquet(self.file_path, columns=columns)
        else:
            raise ValueError(f"Unsupported file format: {self.file_path}")

        self.protein_encoder.fit(df[self.protein_col].unique())

        # Select reference molecules (prioritize known binders)
        if self.target_col is not None:
            binders = df[df[self.target_col] == 1][self.smiles_col].unique()
            non_binders = df[df[self.target_col] == 0][self.smiles_col].unique()

            n_binders = min(len(binders), self.max_ref_mols // 2)
            n_non_binders = min(len(non_binders), self.max_ref_mols - n_binders)

            reference_smiles = np.random.choice(binders, n_binders, replace=False).tolist() + \
                               np.random.choice(non_binders, n_non_binders, replace=False).tolist()
        else:
            # If there's no target column, just select random molecules
            all_smiles = df[self.smiles_col].unique()
            reference_smiles = np.random.choice(all_smiles, min(len(all_smiles), self.max_ref_mols),
                                                replace=False).tolist()

        # Calculate reference fingerprints
        for smiles in reference_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                self.reference_fps[smiles] = GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)

    def __iter__(self):
        """
        Iterate over the dataset, yielding processed Data objects.

        Yields:
            Data: A PyTorch Geometric Data object containing molecule information.
        """
        if self.file_path.endswith('.csv'):
            chunks = pd.read_csv(self.file_path, chunksize=self.chunk_size)
        elif self.file_path.endswith('.parquet'):
            columns = [self.protein_col, self.smiles_col]
            if self.target_col is not None:
                columns.append(self.target_col)
            chunks = pd.read_parquet(self.file_path, columns=columns, chunksize=self.chunk_size)
        else:
            raise ValueError(f"Unsupported file format: {self.file_path}")

        for chunk in chunks:
            for _, row in chunk.iterrows():
                mol = Chem.MolFromSmiles(row[self.smiles_col])
                if mol is None:
                    continue

                x, edge_index, global_features = self._get_molecule_features(mol)

                # Ensure global_features has the correct shape (1, num_features)
                global_features = global_features.unsqueeze(0)

                data = Data(x=x, edge_index=edge_index, global_features=global_features)

                if self.target_col is not None and self.target_col in row:
                    data.y = torch.tensor([row[self.target_col]], dtype=torch.float32)

                data.protein = torch.tensor([self.protein_encoder.transform([row[self.protein_col]])[0]],
                                            dtype=torch.long)

                yield data

    def _get_molecule_features(self, mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate molecule features including atom features, LogP, and Tanimoto similarity.

        Args:
            mol (Chem.Mol): RDKit molecule object.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Node features, edge index, and global features tensors.
        """
        # Calculate LogP
        logp = Crippen.MolLogP(mol)

        # Calculate Tanimoto similarities
        mol_fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        tanimoto_sims = [DataStructs.TanimotoSimilarity(mol_fp, ref_fp) for ref_fp in self.reference_fps.values()]

        # Calculate other molecular features
        mol_features = [
            CalcNumRings(mol),
            AllChem.CalcNumRotatableBonds(mol),
            AllChem.CalcNumHBD(mol),
            AllChem.CalcNumHBA(mol),
            CalcNumAmideBonds(mol),
            CalcNumSpiroAtoms(mol),
            ExactMolWt(mol),
            MolWt(mol),
            HeavyAtomMolWt(mol),
            MaxAbsPartialCharge(mol),
            MinAbsPartialCharge(mol),
            NumRadicalElectrons(mol),
            NumValenceElectrons(mol),
        ]

        # Node features
        node_features = []
        for atom in mol.GetAtoms():
            atom_features = [
                atom.GetAtomicNum(),
                atom.GetTotalDegree(),
                atom.GetFormalCharge(),
                atom.GetTotalNumHs(),
                atom.GetNumRadicalElectrons(),
                atom.GetHybridization(),
                atom.GetIsAromatic(),
                atom.IsInRing(),
            ]
            node_features.append(atom_features + mol_features)

        x = torch.tensor(node_features, dtype=torch.float)

        # Global features
        global_features = torch.tensor([logp] + tanimoto_sims, dtype=torch.float)

        # Calculate edge index
        edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges += [[i, j], [j, i]]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return x, edge_index, global_features


class NaiveFourierKANLayer(nn.Module):
    """
    Naive Fourier Kolmogorov-Arnold Network Layer.

    This layer applies a Fourier transform to the input and uses learnable coefficients
    to create a complex, non-linear transformation.
    """

    def __init__(self, inputdim: int, outdim: int, gridsize: int = 300, addbias: bool = True):
        """
        Initialize the NaiveFourierKANLayer.

        Args:
            inputdim (int): Dimension of the input features.
            outdim (int): Dimension of the output features.
            gridsize (int): Size of the Fourier grid. Defaults to 300.
            addbias (bool): Whether to add a bias term. Defaults to True.
        """
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        # Initialize Fourier coefficients
        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, outdim, inputdim, gridsize) /
            (torch.sqrt(torch.tensor(inputdim).float()) * torch.sqrt(torch.tensor(self.gridsize).float()))
        )
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NaiveFourierKANLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, inputdim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, outdim).
        """
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)

        # Create frequency grid
        k = torch.arange(1, self.gridsize + 1, device=x.device).view(1, 1, 1, self.gridsize)
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)

        # Compute Fourier features
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        c = c.view(1, x.shape[0], x.shape[1], self.gridsize)
        s = s.view(1, x.shape[0], x.shape[1], self.gridsize)

        # Apply Fourier coefficients
        y = torch.einsum("dbik,djik->bj", torch.cat([c, s], dim=0), self.fouriercoeffs)

        if self.addbias:
            y += self.bias

        return y.view(outshape)


class BELKAModule(pl.LightningModule):
    def __init__(
            self,
            in_feat: int,
            hidden_feat: int,
            out_feat: int,
            num_layers: int,
            num_proteins: int,
            protein_embedding_dim: int,
            global_feat: int,
            learning_rate: float,
            grid_feat: int = 200
    ):
        super().__init__()
        self.protein_embedding = nn.Embedding(num_proteins, protein_embedding_dim)

        self.convs = nn.ModuleList()
        self.kans = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(in_feat, hidden_feat))
            else:
                self.convs.append(GCNConv(hidden_feat, hidden_feat))
            self.kans.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat))

        self.global_attention = nn.Sequential(
            nn.Linear(global_feat, hidden_feat),
            nn.ReLU(),
            nn.Linear(hidden_feat, 1)
        )

        self.final_layer = nn.Linear(hidden_feat + protein_embedding_dim + global_feat, out_feat)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([99.0]))
        self.learning_rate = learning_rate

    def forward(self, data: Union[Data, Batch]) -> torch.Tensor:
        device = next(self.parameters()).device
        x, edge_index, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
        global_features = data.global_features.to(device)

        # Handle zero nodes case
        if x.size(0) == 0:
            return torch.zeros((1, 1), device=device)

        # Graph convolutions and KAN layers
        for conv, kan in zip(self.convs, self.kans):
            x = conv(x, edge_index)
            x = kan(x)
            x = F.relu(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Attention on global features
        attention_weights = self.global_attention(global_features).squeeze(-1)
        attention_weights = F.softmax(attention_weights, dim=0)
        attended_global = torch.sum(global_features * attention_weights.unsqueeze(-1), dim=0)

        # Combine with protein embedding
        protein_emb = self.protein_embedding(data.protein.to(device))
        if len(protein_emb.shape) > 2:
            protein_emb = protein_emb.mean(dim=1)  # Average protein embeddings if multiple per graph
        elif len(protein_emb.shape) == 1:
            protein_emb = protein_emb.unsqueeze(0)  # Add batch dimension if single protein

        # Ensure all tensors have the same batch size
        num_graphs = x.size(0)
        protein_emb = protein_emb.expand(num_graphs, -1)
        attended_global = attended_global.unsqueeze(0).expand(num_graphs, -1)

        x = torch.cat([x, protein_emb, attended_global], dim=1)

        return self.final_layer(x)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.criterion(y_hat, batch.y.to(y_hat.device))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                               verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


class BELKADataModule(pl.LightningDataModule):
    def __init__(self, train_file: str, test_file: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = StreamingMoleculeDataset(self.train_file)
            self.val_dataset = StreamingMoleculeDataset(self.train_file)  # Use a separate iterator for validation
        if stage == 'test' or stage is None:
            self.test_dataset = StreamingMoleculeDataset(self.test_file, target_col=None)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True)


def main():
    pl.seed_everything(42)
    train_file = "kaggle/input/leash-BELKA/train.csv"
    test_file = "kaggle/input/leash-BELKA/test.parquet"

    data_module = BELKADataModule(train_file, test_file, batch_size=128, num_workers=4)
    data_module.setup()

    # Get a sample batch to determine input dimensions
    sample_batch = next(iter(data_module.train_dataloader()))
    in_feat = sample_batch.x.size(-1)
    num_proteins = len(data_module.train_dataset.protein_encoder.classes_)
    global_feat = sample_batch.global_features.size(-1)

    model = BELKAModule(
        in_feat=in_feat,
        hidden_feat=256,
        out_feat=1,
        num_layers=2,
        num_proteins=num_proteins,
        protein_embedding_dim=32,
        global_feat=global_feat,
        learning_rate=1e-3,
        grid_feat=200
    )

    logger.info("🏗️ Model architecture:")
    logger.info(model)

    callbacks = [
        EarlyStopping(monitor='val_ap', patience=10, mode='max'),
        ModelCheckpoint(monitor='val_ap', save_top_k=3, mode='max', filename='belka-{epoch:02d}-{val_ap:.4f}'),
        LearningRateMonitor(logging_interval='step')
    ]

    mlf_logger = MLFlowLogger(experiment_name="BELKA_GKAN", tracking_uri="mlruns")

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=callbacks,
        logger=mlf_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision="16-mixed",
        gradient_clip_val=0.5,
        accumulate_grad_batches=4,
        log_every_n_steps=50,
        num_sanity_val_steps=0
    )

    logger.info("🚂 Starting model training")
    trainer.fit(model, data_module)

    logger.info("🔮 Making predictions on test set")
    predictions = trainer.predict(model, data_module.test_dataloader())
    predictions = torch.cat([pred for batch in predictions for pred in batch]).cpu().numpy()

    sample_submission = pd.read_csv("kaggle/input/leash-BELKA/sample_submission.csv")
    sample_submission['binds'] = predictions
    sample_submission.to_csv('submission.csv', index=False)
    logger.info("💾 Submission saved to submission.csv")


if __name__ == "__main__":
    main()
