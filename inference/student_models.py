"""Inference-only definitions matching the saved 80:20 Full Student architectures.

Layer names, normalization, residual blocks and head scales match the original
04_protein_student.py / 05_dna_student.py. No optimizer or training code here.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CosineHead(nn.Module):
    def __init__(self, number_classes, dimension, scale):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(number_classes, dimension))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale)

    def forward(self, features):
        return self.scale * F.linear(F.normalize(features, p=2, dim=1),
                                     F.normalize(self.weight, p=2, dim=1))


class ResidualBlock(nn.Module):
    def __init__(self, dimension, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, dimension), nn.LayerNorm(dimension),
                                 nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dimension, dimension), nn.LayerNorm(dimension))

    def forward(self, x):
        return F.gelu(x + self.net(x))


class ProteinStudent(nn.Module):
    def __init__(self, protein_dimension, protein_mean, protein_std, number_arg_subtypes, config):
        super().__init__()
        self.register_buffer("protein_mean", protein_mean.clone())
        self.register_buffer("protein_std", protein_std.clone())
        self.input = nn.Sequential(
            nn.Linear(protein_dimension, config.protein_hidden_1), nn.LayerNorm(config.protein_hidden_1),
            nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(config.protein_hidden_1, config.protein_hidden_2), nn.LayerNorm(config.protein_hidden_2),
            nn.GELU(), nn.Dropout(config.dropout))
        self.blocks = nn.Sequential(ResidualBlock(config.protein_hidden_2, config.dropout),
                                    ResidualBlock(config.protein_hidden_2, config.dropout))
        self.output = nn.Sequential(nn.Linear(config.protein_hidden_2, config.embedding_dim),
                                    nn.LayerNorm(config.embedding_dim))
        self.binary_head = CosineHead(2, config.embedding_dim, config.logit_scale_binary)
        self.arg_head = CosineHead(number_arg_subtypes, config.embedding_dim, config.logit_scale_subtype)

    def encode(self, protein):
        protein = (protein - self.protein_mean) / self.protein_std
        return F.normalize(self.output(self.blocks(self.input(protein))), p=2, dim=1)

    def forward(self, protein):
        z = self.encode(protein)
        return self.binary_head(z), self.arg_head(z), z


class DNAStudent(nn.Module):
    def __init__(self, dna_dimension, dna_mean, dna_std, number_arg_subtypes, config):
        super().__init__()
        self.register_buffer("dna_mean", dna_mean.clone())
        self.register_buffer("dna_std", dna_std.clone())
        self.input = nn.Sequential(nn.Linear(dna_dimension, config.hidden_dim),
                                   nn.LayerNorm(config.hidden_dim), nn.GELU(), nn.Dropout(config.dropout))
        self.blocks = nn.Sequential(ResidualBlock(config.hidden_dim, config.dropout),
                                    ResidualBlock(config.hidden_dim, config.dropout))
        self.output = nn.Sequential(nn.Linear(config.hidden_dim, config.embedding_dim),
                                    nn.LayerNorm(config.embedding_dim))
        self.binary_head = CosineHead(2, config.embedding_dim, config.logit_scale_binary)
        self.arg_head = CosineHead(number_arg_subtypes, config.embedding_dim, config.logit_scale_subtype)

    def encode(self, dna):
        dna = (dna - self.dna_mean) / self.dna_std
        return F.normalize(self.output(self.blocks(self.input(dna))), p=2, dim=1)

    def forward(self, dna):
        z = self.encode(dna)
        return self.binary_head(z), self.arg_head(z), z
