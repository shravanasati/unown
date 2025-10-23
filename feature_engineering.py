"""Feature engineering utilities for the Pokemon legendary classifier."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureBuilder:
    """Constructs domain-informed features for the Pokemon dataset."""

    target_col: str = "is_legendary"
    stat_cols: Sequence[str] = field(
        default_factory=lambda: ("hp", "attack", "defense", "sp_atk", "sp_def", "speed")
    )
    feature_columns: List[str] = field(default_factory=list)

    # Learned statistics
    global_legendary_rate: float = 0.0
    primary_type_rate: Dict[str, float] = field(default_factory=dict)
    secondary_type_rate: Dict[str, float] = field(default_factory=dict)
    type_combo_rate: Dict[str, float] = field(default_factory=dict)
    generation_rate: Dict[int, float] = field(default_factory=dict)
    primary_type_strength: Dict[str, float] = field(default_factory=dict)
    secondary_type_strength: Dict[str, float] = field(default_factory=dict)
    generation_strength: Dict[int, float] = field(default_factory=dict)
    primary_type_frequency: Dict[str, float] = field(default_factory=dict)
    secondary_type_frequency: Dict[str, float] = field(default_factory=dict)
    type_combo_frequency: Dict[str, float] = field(default_factory=dict)
    generation_frequency: Dict[int, float] = field(default_factory=dict)
    ability_rate: Dict[str, float] = field(default_factory=dict)
    ability_frequency: Dict[str, float] = field(default_factory=dict)
    stat_means: Dict[str, float] = field(default_factory=dict)
    stat_stds: Dict[str, float] = field(default_factory=dict)
    base_total_mean: float = 0.0
    base_total_std: float = 1.0
    height_mean: float = 0.0
    height_std: float = 1.0
    weight_mean: float = 0.0
    weight_std: float = 1.0
    base_experience_mean: float = 0.0
    base_experience_std: float = 1.0

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        data = df.copy()
        data = self._base_features(data, include_target=True)

        if data.empty:
            raise ValueError("Cannot fit FeatureBuilder on an empty dataframe")

        if self.target_col not in data.columns:
            raise KeyError(
                f"Target column '{self.target_col}' missing while fitting FeatureBuilder"
            )

        self.global_legendary_rate = float(data[self.target_col].mean())

        # Type-based statistics
        self.primary_type_rate = (
            data.groupby("primary_type")[self.target_col].mean().to_dict()
        )
        self.secondary_type_rate = (
            data.groupby("secondary_type")[self.target_col].mean().to_dict()
        )
        self.type_combo_rate = (
            data.groupby("type_combo")[self.target_col].mean().to_dict()
        )
        self.generation_rate = (
            data.groupby("generation")[self.target_col].mean().to_dict()
        )

        # Frequency based priors
        total_rows = len(data)
        self.primary_type_frequency = (
            data.groupby("primary_type").size().div(total_rows).to_dict()
        )
        self.secondary_type_frequency = (
            data.groupby("secondary_type").size().div(total_rows).to_dict()
        )
        self.type_combo_frequency = (
            data.groupby("type_combo").size().div(total_rows).to_dict()
        )
        self.generation_frequency = (
            data.groupby("generation").size().div(total_rows).to_dict()
        )

        # Strength estimates (average base stats by grouping)
        self.primary_type_strength = (
            data.groupby("primary_type")["base_total"].mean().to_dict()
        )
        self.secondary_type_strength = (
            data.groupby("secondary_type")["base_total"].mean().to_dict()
        )
        self.generation_strength = (
            data.groupby("generation")["base_total"].mean().to_dict()
        )

        # Ability statistics
        ability_sum = defaultdict(float)
        ability_count = defaultdict(int)
        ability_freq = defaultdict(int)
        for abilities, is_legendary in zip(
            data["_abilities_list"], data[self.target_col]
        ):
            if not abilities:
                continue
            for ability in abilities:
                ability_sum[ability] += float(is_legendary)
                ability_count[ability] += 1
                ability_freq[ability] += 1

        self.ability_rate = {
            ability: ability_sum[ability] / ability_count[ability]
            for ability in ability_sum
            if ability_count[ability] > 0
        }
        self.ability_frequency = {
            ability: ability_freq[ability] / total_rows for ability in ability_freq
        }

        # Stat distributions for z-scores
        self.stat_means = {col: float(data[col].mean()) for col in self.stat_cols}
        self.stat_stds = {
            col: float(max(data[col].std(), 1e-6)) for col in self.stat_cols
        }

        self.base_total_mean = float(data["base_total"].mean())
        self.base_total_std = float(max(data["base_total"].std(), 1e-6))
        self.height_mean = float(data["height"].mean())
        self.height_std = float(max(data["height"].std(), 1e-6))
        self.weight_mean = float(data["weight"].mean())
        self.weight_std = float(max(data["weight"].std(), 1e-6))
        self.base_experience_mean = float(data["base_experience"].mean())
        self.base_experience_std = float(max(data["base_experience"].std(), 1e-6))

        if not self.feature_columns:
            self.feature_columns = self._default_feature_list()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_columns:
            raise RuntimeError(
                "FeatureBuilder must be fitted before calling transform()"
            )

        data = df.copy()
        data = self._base_features(data, include_target=False)

        # Apply learned statistics with fallbacks
        fallback_rate = self.global_legendary_rate

        data["primary_type_legendary_rate"] = (
            data["primary_type"].map(self.primary_type_rate).fillna(fallback_rate)
        )
        data["secondary_type_legendary_rate"] = (
            data["secondary_type"].map(self.secondary_type_rate).fillna(fallback_rate)
        )
        data["type_combo_legendary_rate"] = (
            data["type_combo"].map(self.type_combo_rate).fillna(fallback_rate)
        )
        data["generation_legendary_rate"] = (
            data["generation"].map(self.generation_rate).fillna(fallback_rate)
        )

        data["primary_type_strength"] = (
            data["primary_type"]
            .map(self.primary_type_strength)
            .fillna(self.base_total_mean)
        )
        data["secondary_type_strength"] = (
            data["secondary_type"]
            .map(self.secondary_type_strength)
            .fillna(self.base_total_mean)
        )
        data["generation_strength"] = (
            data["generation"]
            .map(self.generation_strength)
            .fillna(self.base_total_mean)
        )

        data["primary_type_frequency"] = (
            data["primary_type"].map(self.primary_type_frequency).fillna(0.0)
        )
        data["secondary_type_frequency"] = (
            data["secondary_type"].map(self.secondary_type_frequency).fillna(0.0)
        )
        data["type_combo_frequency"] = (
            data["type_combo"].map(self.type_combo_frequency).fillna(0.0)
        )
        data["generation_frequency"] = (
            data["generation"].map(self.generation_frequency).fillna(0.0)
        )

        ability_stats = data["_abilities_list"].apply(self._ability_rate_features)
        data[
            [
                "ability_mean_legendary_rate",
                "ability_max_legendary_rate",
                "ability_frequency_mean",
                "ability_frequency_max",
            ]
        ] = pd.DataFrame(ability_stats.tolist(), index=data.index)

        for col in self.stat_cols:
            mean = self.stat_means.get(col, 0.0)
            std = self.stat_stds.get(col, 1.0)
            data[f"{col}_zscore"] = (data[col] - mean) / std

        data["base_total_zscore"] = (
            data["base_total"] - self.base_total_mean
        ) / self.base_total_std
        data["height_zscore"] = (data["height"] - self.height_mean) / self.height_std
        data["weight_zscore"] = (data["weight"] - self.weight_mean) / self.weight_std
        data["base_experience_zscore"] = (
            data["base_experience"] - self.base_experience_mean
        ) / self.base_experience_std

        # Drop helper columns not meant for downstream models
        data = data.drop(columns=["_parsed_types", "_abilities_list"], errors="ignore")

        return data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _base_features(self, df: pd.DataFrame, include_target: bool) -> pd.DataFrame:
        data = df.copy()

        if "types" not in data.columns:
            data["types"] = "unknown"

        data["types"] = data["types"].fillna("unknown")
        data["_parsed_types"] = data["types"].apply(self._parse_types)
        data["primary_type"] = data["_parsed_types"].apply(lambda x: x[0])
        data["secondary_type"] = data["_parsed_types"].apply(
            lambda x: x[1] if len(x) > 1 else "none"
        )
        data["type_count"] = data["_parsed_types"].apply(len)
        data["is_dual_type"] = (data["type_count"] > 1).astype(int)
        data["type_combo"] = data["primary_type"] + "|" + data["secondary_type"]

        data["abilities"] = data.get("abilities", "").fillna("")
        data["_abilities_list"] = data["abilities"].apply(self._parse_abilities)
        data["abilities_count"] = data["_abilities_list"].apply(len)
        data["unique_abilities_count"] = data["_abilities_list"].apply(
            lambda x: len(set(x))
        )
        data["has_hidden_ability"] = (data["abilities_count"] > 2).astype(int)

        # Ensure numeric columns exist
        numeric_defaults = {
            "hp": 0,
            "attack": 0,
            "defense": 0,
            "sp_atk": 0,
            "sp_def": 0,
            "speed": 0,
            "base_total": 0,
            "height": 0.0,
            "weight": 0.0,
            "base_experience": 0,
            "generation": 0,
        }
        for col, default in numeric_defaults.items():
            if col not in data.columns:
                data[col] = default
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(default)

        # Ensure continuous features are floats for downstream numpy ops
        float_columns = [
            "height",
            "weight",
            "base_total",
            "base_experience",
            "hp",
            "attack",
            "defense",
            "sp_atk",
            "sp_def",
            "speed",
        ]
        for col in float_columns:
            if col in data.columns:
                data[col] = data[col].astype(float)

        data["attack_defense_ratio"] = data["attack"] / (data["defense"] + 1)
        data["attack_speed_ratio"] = data["attack"] / (data["speed"] + 1)
        data["special_attack_ratio"] = data["sp_atk"] / (data["sp_def"] + 1)
        data["attack_spatk_ratio"] = data["attack"] / (data["sp_atk"] + 1)
        data["defense_spdef_ratio"] = data["defense"] / (data["sp_def"] + 1)

        data["physical_total"] = data["attack"] + data["defense"] + data["hp"]
        data["special_total"] = data["sp_atk"] + data["sp_def"] + data["hp"]
        data["offense_total"] = data["attack"] + data["sp_atk"]
        data["defense_total"] = data["defense"] + data["sp_def"]
        data["offense_defense_gap"] = data["offense_total"] - data["defense_total"]
        data["bulk_index"] = (data["defense"] + data["sp_def"] + data["hp"]) / 3
        data["hyper_offense_index"] = (
            data["attack"] + data["sp_atk"] + data["speed"]
        ) / 3

        stats_frame = data[list(self.stat_cols)]
        data["stat_mean"] = stats_frame.mean(axis=1)
        data["stat_std"] = stats_frame.std(axis=1)
        data["stat_min"] = stats_frame.min(axis=1)
        data["stat_max"] = stats_frame.max(axis=1)
        data["stat_cv"] = data["stat_std"] / (data["stat_mean"] + 1e-6)

        data["speed_base_total_ratio"] = data["speed"] / (data["base_total"] + 1)
        data["speed_offense_ratio"] = data["speed"] / (data["offense_total"] + 1)

        data["log_height"] = np.log1p(data["height"].clip(lower=0))
        data["log_weight"] = np.log1p(data["weight"].clip(lower=0))
        data["log_base_experience"] = np.log1p(data["base_experience"].clip(lower=0))

        data["height_weight_ratio"] = data["height"] / (data["weight"] + 1)
        data["bmi_like"] = data["weight"] / (np.square(data["height"]) + 1)

        # Ability frequency-derived features (fallback to zeros, actual values filled after fit)
        data["ability_frequency_mean"] = 0.0
        data["ability_frequency_max"] = 0.0

        if include_target and self.target_col not in data.columns:
            raise KeyError(f"Target column '{self.target_col}' missing from dataframe")

        return data

    def _parse_types(self, raw: str) -> List[str]:
        if not isinstance(raw, str):
            return ["unknown"]
        parts = [part.strip().lower() for part in raw.split(",") if part.strip()]
        return parts or ["unknown"]

    def _parse_abilities(self, raw: str) -> List[str]:
        if not isinstance(raw, str):
            return []
        parts = [part.strip().lower() for part in raw.split(";") if part.strip()]
        return parts

    def _ability_rate_features(
        self, abilities: Iterable[str]
    ) -> Tuple[float, float, float, float]:
        abilities = list(abilities)
        if not abilities:
            default = self.global_legendary_rate
            return default, default, 0.0, 0.0

        rates = [
            self.ability_rate.get(ability, self.global_legendary_rate)
            for ability in abilities
        ]
        freqs = [self.ability_frequency.get(ability, 0.0) for ability in abilities]
        return (
            float(np.mean(rates)),
            float(np.max(rates)),
            float(np.mean(freqs)) if freqs else 0.0,
            float(np.max(freqs)) if freqs else 0.0,
        )

    def _default_feature_list(self) -> List[str]:
        return [
            "generation",
            "hp",
            "attack",
            "defense",
            "sp_atk",
            "sp_def",
            "speed",
            "base_total",
            "height",
            "weight",
            "base_experience",
            "attack_defense_ratio",
            "attack_speed_ratio",
            "special_attack_ratio",
            "physical_total",
            "special_total",
            "type_count",
            "is_dual_type",
            "primary_type_legendary_rate",
            "secondary_type_legendary_rate",
            "type_combo_legendary_rate",
            "generation_legendary_rate",
            "primary_type_strength",
            "secondary_type_strength",
            "generation_strength",
            "primary_type_frequency",
            "secondary_type_frequency",
            "type_combo_frequency",
            "generation_frequency",
            "log_height",
            "log_weight",
            "log_base_experience",
            "stat_mean",
            "stat_std",
            "stat_cv",
            "stat_max",
            "stat_min",
            "offense_total",
            "defense_total",
            "offense_defense_gap",
            "speed_base_total_ratio",
            "speed_offense_ratio",
            "attack_spatk_ratio",
            "defense_spdef_ratio",
            "bulk_index",
            "hyper_offense_index",
            "height_weight_ratio",
            "bmi_like",
            "abilities_count",
            "unique_abilities_count",
            "has_hidden_ability",
            "ability_mean_legendary_rate",
            "ability_max_legendary_rate",
            "ability_frequency_mean",
            "ability_frequency_max",
            "hp_zscore",
            "attack_zscore",
            "defense_zscore",
            "sp_atk_zscore",
            "sp_def_zscore",
            "speed_zscore",
            "base_total_zscore",
            "height_zscore",
            "weight_zscore",
            "base_experience_zscore",
        ]
