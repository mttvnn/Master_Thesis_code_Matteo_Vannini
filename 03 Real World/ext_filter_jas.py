"""Standalone JAS sit-to-stand/stand-to-sit detector.

Note on filter_transitions: this is a post-processing step that applies advanced heuristics to remove
false positives from the detected transitions. It is built into the detector: ``JASDetector(filter_transitions=
"default")`` (or "impaired", the other threshold table

"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ORIGINAL_DEFAULT_THRESHOLDS = {
    'sit2stand': {
        'valid_angle_range': (20, 50),
        'min_ext_vel': 12,
        'min_disp': 0.15,
        'high_disp': 0.255,
        'high_flex_vel': 70,
        'max_allowed_ext_vel': 60,
        'max_allowed_flex_vel': 90,
        'min_allowed_ratio': 1.05
    },
    'stand2sit': {
        'valid_angle_range': (6, 60),
        'min_angle': 18,
        'min_flex_vel': 12,
        'min_disp': 0.15,
        'high_disp': 0.2,
        'high_flex_vel': 60,
        'brake_high_ext': 70,
        'brake_mod_ext': 55,
        'brake_mod_disp': 0.10,
        'brake_mod_angle': 12
    }
}


PFF_THRESHOLDS = {
    'sit2stand': {
        'valid_angle_range': (20, 20),  # 50 default, in PFF control needed every time to avoid too much FP >50 no longer blocks; it triggers the FP heuristic checks
        'min_ext_vel': 12,
        'min_disp': 0.15,
        'high_disp': 0.255,
        'high_flex_vel': 70,
        'max_allowed_ext_vel': 80, #60 defualt
        'max_allowed_flex_vel': 90,
        'min_allowed_ratio':0.95 #default 1.05
    },
    'stand2sit': {
        'valid_angle_range': (6, 60),
        'min_angle': 18,
        'min_flex_vel': 12,
        'min_disp': 0.15,
        'high_disp': 0.2,
        'high_flex_vel': 60,
        'brake_high_ext': 70,
        'brake_mod_ext': 55,
        'brake_mod_disp': 0.10,
        'brake_mod_angle': 12
    }
}

IMPAIRED_THRESHOLDS = {
    'sit2stand': {
        'valid_angle_range': (10, 60),  # >60 triggers the FP heuristic checks for impaired users
        'min_ext_vel': 9,
        'min_disp': 0.10,
        'high_disp': 0.20,
        'high_flex_vel': 50,
        'max_allowed_ext_vel': 80,
        'max_allowed_flex_vel': 110,
        'min_allowed_ratio': 0.95
    },
    'stand2sit': {
        'valid_angle_range': (4, 60),
        'min_angle': 12,
        'min_flex_vel': 8,
        'min_disp': 0.10,
        'high_disp': 0.20,
        'high_flex_vel': 45,
        'brake_high_ext': 50,
        'brake_mod_ext': 40,
        'brake_mod_disp': 0.08,
        'brake_mod_angle': 8
    }
}

# <<< If USE_PFF = True --> with default it uses PFF threshold
USE_PFF = True
DEFAULT_THRESHOLDS = PFF_THRESHOLDS if USE_PFF else ORIGINAL_DEFAULT_THRESHOLDS

def filter_rules(df: pd.DataFrame, cfg: dict = DEFAULT_THRESHOLDS) -> pd.DataFrame:
    """
    The rules of the transition filter that each row of ``df`` fails: one boolean column per name in FILTER_RULES.
    A sit2stand or stand2sit row passes the filter when it fails none; rows of other types fail none.
    High angles are no longer hard-blocked, but checked for dynamic false positives.
    """
    s2st_cfg = cfg['sit2stand']
    st2s_cfg = cfg['stand2sit']
    tilt = df['sts_tilt_angle']
    flexion = df['sts_max_flexion_vel']
    extension = df['sts_max_extension_vel']
    displacement = df['sts_vertical_displacement']
    is_s2st = df['transition_type'] == 'sit2stand'
    is_st2s = df['transition_type'] == 'stand2sit'

    # ==========================================
    # --- 1. SIT-TO-STAND (sit2stand) ---------
    # ==========================================
    # Unpack boundaries: standard minimum floor and the conditional heuristic trigger ceiling
    min_angle, h_angle_trigger = s2st_cfg['valid_angle_range']

    # Calculate velocity ratio (flex / ext) safely
    ratio = flexion / extension.replace(0, np.nan)

    # Heuristic Rule: Identify false positives ONLY when angle > trigger ceiling
    is_extreme_movement = (extension > s2st_cfg['max_allowed_ext_vel']) & (flexion > s2st_cfg['max_allowed_flex_vel'])
    is_low_ratio = ratio < s2st_cfg['min_allowed_ratio']
    s2st_false_positive = (tilt > h_angle_trigger) & (is_extreme_movement | is_low_ratio)

    # Standard kinematic and displacement checks
    s2st_cond_ext = extension > s2st_cfg['min_ext_vel']
    s2st_cond_disp = (displacement > s2st_cfg['min_disp']) & (
            (displacement > s2st_cfg['high_disp']) | (flexion > s2st_cfg['high_flex_vel'])
    )

    # ==========================================
    # --- 2. STAND-TO-SIT (stand2sit) ---------
    # ==========================================
    # Stand-to-sit uses the valid range as a strict bounding box
    min_r, max_r = st2s_cfg['valid_angle_range']
    st2s_cond_valid_range = (tilt >= min_r) & (tilt <= max_r)

    st2s_cond_angle = tilt > st2s_cfg['min_angle']
    st2s_cond_flex = flexion > st2s_cfg['min_flex_vel']
    st2s_cond_disp = (displacement.abs() > st2s_cfg['min_disp']) & (
            (displacement.abs() > st2s_cfg['high_disp']) | (flexion > st2s_cfg['high_flex_vel'])
    )
    st2s_base_condition = st2s_cond_angle & st2s_cond_flex & st2s_cond_disp

    st2s_brake_high = extension > st2s_cfg['brake_high_ext']
    st2s_brake_moderate = (extension > st2s_cfg['brake_mod_ext']) & (
            (displacement.abs() > st2s_cfg['brake_mod_disp']) | (tilt > st2s_cfg['brake_mod_angle'])
    )
    st2s_cond_braking = st2s_brake_high | st2s_brake_moderate

    # A sit2stand is kept above the angle floor, without the high-angle false positive, and with the kinematic and
    # displacement checks; a stand2sit inside the angle range with the base condition or braking
    return pd.DataFrame({
        'angle_floor': is_s2st & ~(tilt >= min_angle),
        'high_angle': is_s2st & s2st_false_positive,
        'extension_velocity': is_s2st & ~s2st_cond_ext,
        'displacement': is_s2st & ~s2st_cond_disp,
        'angle_range': is_st2s & ~st2s_cond_valid_range,
        'base_or_braking': is_st2s & ~(st2s_base_condition | st2s_cond_braking),
    }, index=df.index)


def filter_transitions(df: pd.DataFrame, cfg: dict = DEFAULT_THRESHOLDS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Optimized filter for sit2stand and stand2sit transitions using advanced population heuristics.
    High angles are no longer hard-blocked, but checked for dynamic false positives.
    Returns the kept sit2stand and stand2sit rows, then the removed ones: a row is removed when it fails any rule of
    filter_rules(df, cfg).
    """
    failed = filter_rules(df, cfg).any(axis=1)
    is_s2st = df['transition_type'] == 'sit2stand'
    is_st2s = df['transition_type'] == 'stand2sit'
    df_sit2stand = df.loc[is_s2st & ~failed].reset_index(drop=True)
    df_stand2sit = df.loc[is_st2s & ~failed].reset_index(drop=True)
    df_sit2stand_removed = df.loc[is_s2st & failed].reset_index(drop=True)
    df_stand2sit_removed = df.loc[is_st2s & failed].reset_index(drop=True)
    return df_sit2stand, df_stand2sit, df_sit2stand_removed, df_stand2sit_removed

