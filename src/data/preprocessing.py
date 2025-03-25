"""
データ前処理モジュール

JVDデータベースから抽出したデータのクレンジングや型変換を行う関数を提供します。
"""
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime

# ロガーの設定
logger = logging.getLogger(__name__)


def clean_race_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    レース基本情報（jvd_ra）のクレンジングと型変換を行う
    
    Args:
        df: レース基本情報DataFrame
        
    Returns:
        pd.DataFrame: クレンジング・型変換済みのDataFrame
    """
    # コピーを作成
    df_cleaned = df.copy()
    
    # 各カラムのクレンジングと型変換
    # kaisai_nen, kaisai_tsukihi → race_date（日付型）
    try:
        df_cleaned['race_date'] = pd.to_datetime(
            df_cleaned['kaisai_nen'] + df_cleaned['kaisai_tsukihi'],
            format='%Y%m%d',
            errors='coerce'
        )
    except Exception as e:
        logger.warning(f"日付変換でエラーが発生しました: {e}")
    
    # kyori（距離）を数値型に変換
    try:
        df_cleaned['kyori'] = pd.to_numeric(df_cleaned['kyori'], errors='coerce')
    except Exception as e:
        logger.warning(f"距離変換でエラーが発生しました: {e}")
    
    # コース種別を追加
    df_cleaned['course_type'] = df_cleaned['track_code'].apply(
        lambda x: '芝' if str(x).startswith('1') else 'ダート' if str(x).startswith('2') else 'その他'
    )
    
    # 天候コードを名称に変換
    weather_code_map = {
        '1': '晴',
        '2': '曇',
        '3': '小雨',
        '4': '雨',
        '5': '小雪',
        '6': '雪',
        '7': '霧'
    }
    df_cleaned['weather'] = df_cleaned['tenko_code'].map(
        lambda x: weather_code_map.get(str(x), '不明')
    )
    
    # 馬場状態コードを名称に変換
    track_condition_map = {
        '1': '良',
        '2': '稍重',
        '3': '重',
        '4': '不良'
    }
    df_cleaned['track_condition'] = df_cleaned['babajotai_code'].map(
        lambda x: track_condition_map.get(str(x), '不明')
    )
    
    # 競馬場コードを名称に変換
    course_code_map = {
        '01': '札幌',
        '02': '函館',
        '03': '福島',
        '04': '新潟',
        '05': '東京',
        '06': '中山',
        '07': '中京',
        '08': '京都',
        '09': '阪神',
        '10': '小倉'
    }
    df_cleaned['course_name'] = df_cleaned['keibajo_code'].map(
        lambda x: course_code_map.get(str(x), '不明')
    )
    
    # グレードコードを名称に変換
    grade_code_map = {
        'A': 'G1',
        'B': 'G2',
        'C': 'G3',
        'D': 'G', # 重賞
        'E': 'OP', # オープン
        'F': '3勝', # 3勝クラス
        'G': '2勝', # 2勝クラス
        'H': '1勝', # 1勝クラス
        'L': '新馬', # 新馬戦
        'J': '未勝利', # 未勝利
    }
    df_cleaned['grade'] = df_cleaned['grade_code'].map(
        lambda x: grade_code_map.get(str(x), '一般')
    )
    
    # 距離カテゴリを追加
    df_cleaned['distance_category'] = pd.cut(
        df_cleaned['kyori'],
        bins=[0, 1400, 2000, 3000, 5000],
        labels=['短距離', '中距離', '長距離', '超長距離'],
        right=False
    )
    
    # レース年、月を追加
    try:
        df_cleaned['race_year'] = df_cleaned['race_date'].dt.year
        df_cleaned['race_month'] = df_cleaned['race_date'].dt.month
        
        # 季節を追加
        season_map = {
            12: '冬', 1: '冬', 2: '冬',
            3: '春', 4: '春', 5: '春',
            6: '夏', 7: '夏', 8: '夏',
            9: '秋', 10: '秋', 11: '秋'
        }
        df_cleaned['season'] = df_cleaned['race_month'].map(season_map)
    except Exception as e:
        logger.warning(f"日付派生項目の生成でエラーが発生しました: {e}")
    
    return df_cleaned


def clean_horse_race_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    出走馬情報（jvd_se結合データ）のクレンジングと型変換を行う
    
    Args:
        df: 出走馬情報DataFrame
        
    Returns:
        pd.DataFrame: クレンジング・型変換済みのDataFrame
    """
    # コピーを作成
    df_cleaned = df.copy()
    
    # 各カラムのクレンジングと型変換
    # kaisai_nen, kaisai_tsukihi → race_date（日付型）
    try:
        df_cleaned['race_date'] = pd.to_datetime(
            df_cleaned['kaisai_nen'] + df_cleaned['kaisai_tsukihi'],
            format='%Y%m%d',
            errors='coerce'
        )
    except Exception as e:
        logger.warning(f"日付変換でエラーが発生しました: {e}")
    
    # レース情報の変換（race_dataと同様）
    df_cleaned['course_type'] = df_cleaned['track_code'].apply(
        lambda x: '芝' if str(x).startswith('1') else 'ダート' if str(x).startswith('2') else 'その他'
    )
    
    # weather_code_map、track_condition_map、course_code_mapは前の関数と同じ
    weather_code_map = {
        '1': '晴',
        '2': '曇',
        '3': '小雨',
        '4': '雨',
        '5': '小雪',
        '6': '雪',
        '7': '霧'
    }
    df_cleaned['weather'] = df_cleaned['tenko_code'].map(
        lambda x: weather_code_map.get(str(x), '不明')
    )
    
    track_condition_map = {
        '1': '良',
        '2': '稍重',
        '3': '重',
        '4': '不良'
    }
    df_cleaned['track_condition'] = df_cleaned['babajotai_code'].map(
        lambda x: track_condition_map.get(str(x), '不明')
    )
    
    course_code_map = {
        '01': '札幌',
        '02': '函館',
        '03': '福島',
        '04': '新潟',
        '05': '東京',
        '06': '中山',
        '07': '中京',
        '08': '京都',
        '09': '阪神',
        '10': '小倉'
    }
    df_cleaned['course_name'] = df_cleaned['keibajo_code'].map(
        lambda x: course_code_map.get(str(x), '不明')
    )
    
    # 性別コードの変換
    sex_code_map = {
        '1': '牡',
        '2': '牝',
        '3': 'セ',
        ' ': '不明'
    }
    df_cleaned['sex'] = df_cleaned['seibetsu_code'].map(
        lambda x: sex_code_map.get(str(x), '不明')
    )
    
    # 距離を数値型に変換
    try:
        df_cleaned['kyori'] = pd.to_numeric(df_cleaned['kyori'], errors='coerce')
    except Exception as e:
        logger.warning(f"距離変換でエラーが発生しました: {e}")
    
    # 馬齢を数値型に変換
    try:
        df_cleaned['age'] = pd.to_numeric(df_cleaned['barei'], errors='coerce')
    except Exception as e:
        logger.warning(f"馬齢変換でエラーが発生しました: {e}")
    
    # 枠番・馬番を数値型に変換
    try:
        df_cleaned['gate_number'] = pd.to_numeric(df_cleaned['wakuban'], errors='coerce')
        df_cleaned['horse_number'] = pd.to_numeric(df_cleaned['umaban'], errors='coerce')
    except Exception as e:
        logger.warning(f"枠番・馬番変換でエラーが発生しました: {e}")
    
    # 馬体重・増減を数値型に変換
    try:
        df_cleaned['weight'] = pd.to_numeric(df_cleaned['bataiju'], errors='coerce')
        
        # 増減符号と値を組み合わせて計算
        df_cleaned['weight_diff'] = df_cleaned.apply(
            lambda row: 
                pd.to_numeric(row['zogen_sa'], errors='coerce') * 
                (1 if row['zogen_fugo'] == '+' else -1 if row['zogen_fugo'] == '-' else 0), 
            axis=1
        )
    except Exception as e:
        logger.warning(f"馬体重・増減変換でエラーが発生しました: {e}")
    
    # 着順を数値型に変換
    try:
        # 特殊コード（除外など）を処理
        df_cleaned['finish_pos'] = df_cleaned['kakutei_chakujun'].apply(
            lambda x: np.nan if str(x) in ['00', '99'] or not str(x).isdigit() else int(x)
        )
    except Exception as e:
        logger.warning(f"着順変換でエラーが発生しました: {e}")
    
    # 走破タイム・上がり3Fを数値型に変換
    try:
        # 走破タイム（秒）
        df_cleaned['time_seconds'] = df_cleaned['soha_time'].apply(
            lambda x: np.nan if not str(x).isdigit() else int(x) / 10
        )
        
        # 上がり3F（秒）
        df_cleaned['last_3f_seconds'] = df_cleaned['kohan_3f'].apply(
            lambda x: np.nan if not str(x).isdigit() else int(x) / 10
        )
    except Exception as e:
        logger.warning(f"タイム変換でエラーが発生しました: {e}")
    
    # オッズ・人気を数値型に変換
    try:
        df_cleaned['odds'] = df_cleaned['tansho_odds'].apply(
            lambda x: np.nan if not str(x).isdigit() else int(x) / 10
        )
        
        df_cleaned['popularity'] = pd.to_numeric(df_cleaned['tansho_ninkijun'], errors='coerce')
    except Exception as e:
        logger.warning(f"オッズ・人気変換でエラーが発生しました: {e}")
    
    # 人気カテゴリの追加
    df_cleaned['popularity_category'] = pd.cut(
        df_cleaned['popularity'],
        bins=[0, 3, 8, float('inf')],
        labels=['人気', '中穴', '大穴'],
        right=True
    )
    
    # 距離カテゴリを追加
    df_cleaned['distance_category'] = pd.cut(
        df_cleaned['kyori'],
        bins=[0, 1400, 2000, 3000, 5000],
        labels=['短距離', '中距離', '長距離', '超長距離'],
        right=False
    )
    
    # レース年、月を追加
    try:
        df_cleaned['race_year'] = df_cleaned['race_date'].dt.year
        df_cleaned['race_month'] = df_cleaned['race_date'].dt.month
        
        # 季節を追加
        season_map = {
            12: '冬', 1: '冬', 2: '冬',
            3: '春', 4: '春', 5: '春',
            6: '夏', 7: '夏', 8: '夏',
            9: '秋', 10: '秋', 11: '秋'
        }
        df_cleaned['season'] = df_cleaned['race_month'].map(season_map)
    except Exception as e:
        logger.warning(f"日付派生項目の生成でエラーが発生しました: {e}")
    
    return df_cleaned


def clean_horse_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    馬基本情報（jvd_um）のクレンジングと型変換を行う
    
    Args:
        df: 馬基本情報DataFrame
        
    Returns:
        pd.DataFrame: クレンジング・型変換済みのDataFrame
    """
    # コピーを作成
    df_cleaned = df.copy()
    
    # 各カラムのクレンジングと型変換
    # 生年月日を日付型に変換
    try:
        df_cleaned['birth_date'] = pd.to_datetime(
            df_cleaned['seinengappi'].astype(str),
            format='%Y%m%d',
            errors='coerce'
        )
    except Exception as e:
        logger.warning(f"生年月日変換でエラーが発生しました: {e}")
    
    # 性別コードの変換
    sex_code_map = {
        '1': '牡',
        '2': '牝',
        '3': 'セ'
    }
    df_cleaned['sex'] = df_cleaned['seibetsu_code'].map(
        lambda x: sex_code_map.get(str(x), '不明')
    )
    
    return df_cleaned


def clean_pedigree_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    血統情報のクレンジングと型変換を行う
    
    Args:
        df: 血統情報DataFrame
        
    Returns:
        pd.DataFrame: クレンジング・型変換済みのDataFrame
    """
    # コピーを作成
    df_cleaned = df.copy()
    
    # 繁殖区分を変換
    hanshoku_kubun_map = {
        '01': '種牡馬',
        '02': '繁殖牝馬'
    }
    df_cleaned['hanshoku_type'] = df_cleaned['hanshoku_kubun'].map(
        lambda x: hanshoku_kubun_map.get(str(x), '不明')
    )
    
    # 発入社名を整形
    if 'hatsunyushamei' in df_cleaned.columns:
        df_cleaned['hatsunyushamei'] = df_cleaned['hatsunyushamei'].apply(
            lambda x: str(x).strip() if pd.notna(x) else ''
        )
    
    return df_cleaned


def clean_payouts(df: pd.DataFrame) -> pd.DataFrame:
    """
    払戻情報のクレンジングと型変換を行う
    
    Args:
        df: 払戻情報DataFrame
        
    Returns:
        pd.DataFrame: クレンジング・型変換済みのDataFrame
    """
    # コピーを作成
    df_cleaned = df.copy()
    
    # 各払戻金を数値型に変換
    payout_columns = [col for col in df_cleaned.columns if col.endswith('_money1') 
                      or col.endswith('_money2') or col.endswith('_money3')
                      or col.endswith('_money4') or col.endswith('_money5')
                      or col.endswith('_money6') or col.endswith('_money7')]
    
    for col in payout_columns:
        try:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        except Exception as e:
            logger.warning(f"{col}の変換でエラーが発生しました: {e}")
    
    # kaisai_nen, kaisai_tsukihi → race_date（日付型）
    try:
        df_cleaned['race_date'] = pd.to_datetime(
            df_cleaned['kaisai_nen'] + df_cleaned['kaisai_tsukihi'],
            format='%Y%m%d',
            errors='coerce'
        )
    except Exception as e:
        logger.warning(f"日付変換でエラーが発生しました: {e}")
    
    return df_cleaned


def clean_merged_base_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    結合済みの基本データのクレンジングと型変換を行う
    （horse_race_resultsのクレンジングと重複する部分あり）
    
    Args:
        df: 結合済み基本データDataFrame
        
    Returns:
        pd.DataFrame: クレンジング・型変換済みのDataFrame
    """
    # コピーを作成
    df_cleaned = df.copy()
    
    # 各カラムのクレンジングと型変換
    # kaisai_nen, kaisai_tsukihi → race_date（日付型）
    try:
        df_cleaned['race_date'] = pd.to_datetime(
            df_cleaned['kaisai_nen'] + df_cleaned['kaisai_tsukihi'],
            format='%Y%m%d',
            errors='coerce'
        )
    except Exception as e:
        logger.warning(f"日付変換でエラーが発生しました: {e}")
    
    # レース情報の変換
    df_cleaned['course_type'] = df_cleaned['track_code'].apply(
        lambda x: '芝' if str(x).startswith('1') else 'ダート' if str(x).startswith('2') else 'その他'
    )
    
    # 各種コードマッピング
    weather_code_map = {
        '1': '晴',
        '2': '曇',
        '3': '小雨',
        '4': '雨',
        '5': '小雪',
        '6': '雪',
        '7': '霧'
    }
    df_cleaned['weather'] = df_cleaned['tenko_code'].map(
        lambda x: weather_code_map.get(str(x), '不明')
    )
    
    track_condition_map = {
        '1': '良',
        '2': '稍重',
        '3': '重',
        '4': '不良'
    }
    df_cleaned['track_condition'] = df_cleaned['babajotai_code'].map(
        lambda x: track_condition_map.get(str(x), '不明')
    )
    
    course_code_map = {
        '01': '札幌',
        '02': '函館',
        '03': '福島',
        '04': '新潟',
        '05': '東京',
        '06': '中山',
        '07': '中京',
        '08': '京都',
        '09': '阪神',
        '10': '小倉'
    }
    df_cleaned['course_name'] = df_cleaned['keibajo_code'].map(
        lambda x: course_code_map.get(str(x), '不明')
    )
    
    # 性別コードの変換
    sex_code_map = {
        '1': '牡',
        '2': '牝',
        '3': 'セ'
    }
    df_cleaned['sex'] = df_cleaned['seibetsu_code'].map(
        lambda x: sex_code_map.get(str(x), '不明')
    )
    
    # 数値型への変換
    numeric_conversions = {
        'kyori': 'distance',
        'barei': 'age',
        'wakuban': 'gate_number',
        'umaban': 'horse_number',
        'bataiju': 'weight',
        'kakutei_chakujun': 'finish_pos',
        'soha_time': 'time_seconds',
        'kohan_3f': 'last_3f_seconds',
        'tansho_odds': 'odds',
        'tansho_ninkijun': 'popularity'
    }
    
    for src_col, dst_col in numeric_conversions.items():
        try:
            if src_col in ['kakutei_chakujun']:
                # 特殊コード（除外など）を処理
                df_cleaned[dst_col] = df_cleaned[src_col].apply(
                    lambda x: np.nan if str(x) in ['00', '99'] or not str(x).isdigit() else int(x)
                )
            elif src_col in ['soha_time', 'kohan_3f', 'tansho_odds']:
                # 10分の1の値に変換
                df_cleaned[dst_col] = df_cleaned[src_col].apply(
                    lambda x: np.nan if not str(x).isdigit() else int(x) / 10
                )
            else:
                # 通常の数値変換
                df_cleaned[dst_col] = pd.to_numeric(df_cleaned[src_col], errors='coerce')
        except Exception as e:
            logger.warning(f"{src_col}から{dst_col}への変換でエラーが発生しました: {e}")
    
    # 馬体重増減を計算
    try:
        # 増減符号と値を組み合わせて計算
        df_cleaned['weight_diff'] = df_cleaned.apply(
            lambda row: 
                pd.to_numeric(row['zogen_sa'], errors='coerce') * 
                (1 if row['zogen_fugo'] == '+' else -1 if row['zogen_fugo'] == '-' else 0), 
            axis=1
        )
    except Exception as e:
        logger.warning(f"馬体重増減計算でエラーが発生しました: {e}")
    
    # 人気カテゴリの追加
    df_cleaned['popularity_category'] = pd.cut(
        df_cleaned['popularity'],
        bins=[0, 3, 8, float('inf')],
        labels=['人気', '中穴', '大穴'],
        right=True
    )
    
    # 距離カテゴリを追加
    df_cleaned['distance_category'] = pd.cut(
        df_cleaned['kyori'],
        bins=[0, 1400, 2000, 3000, 5000],
        labels=['短距離', '中距離', '長距離', '超長距離'],
        right=False
    )
    
    # 生年月日を日付型に変換
    try:
        df_cleaned['birth_date'] = pd.to_datetime(
            df_cleaned['seinengappi'].astype(str),
            format='%Y%m%d',
            errors='coerce'
        )
    except Exception as e:
        logger.warning(f"生年月日変換でエラーが発生しました: {e}")
    
    # レース年、月を追加
    try:
        df_cleaned['race_year'] = df_cleaned['race_date'].dt.year
        df_cleaned['race_month'] = df_cleaned['race_date'].dt.month
        
        # 季節を追加
        season_map = {
            12: '冬', 1: '冬', 2: '冬',
            3: '春', 4: '春', 5: '春',
            6: '夏', 7: '夏', 8: '夏',
            9: '秋', 10: '秋', 11: '秋'
        }
        df_cleaned['season'] = df_cleaned['race_month'].map(season_map)
    except Exception as e:
        logger.warning(f"日付派生項目の生成でエラーが発生しました: {e}")
    
    return df_cleaned


def handle_outliers(df: pd.DataFrame, columns: List[str], method: str = 'clip') -> pd.DataFrame:
    """
    外れ値を処理する
    
    Args:
        df: 処理対象のDataFrame
        columns: 外れ値を処理するカラムのリスト
        method: 処理方法（'clip'または'remove'）
        
    Returns:
        pd.DataFrame: 外れ値処理済みのDataFrame
    """
    df_processed = df.copy()
    
    for col in columns:
        if col not in df_processed.columns:
            logger.warning(f"カラム {col} がDataFrameに存在しません")
            continue
        
        # 数値型でない場合はスキップ
        if not pd.api.types.is_numeric_dtype(df_processed[col]):
            logger.warning(f"カラム {col} は数値型ではありません")
            continue
        
        # 外れ値の範囲を計算
        q1 = df_processed[col].quantile(0.01)
        q3 = df_processed[col].quantile(0.99)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # 外れ値の処理
        outliers = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
        
        if method == 'clip':
            # クリッピング（上下限値で置換）
            df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
            df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
            logger.info(f"カラム {col} の外れ値 {outliers.sum()} 件をクリッピングしました")
        
        elif method == 'remove':
            # 行の削除
            df_processed = df_processed[~outliers]
            logger.info(f"カラム {col} の外れ値により {outliers.sum()} 行を削除しました")
    
    return df_processed


def fill_missing_values(df: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
    """
    欠損値を補完する
    
    Args:
        df: 処理対象のDataFrame
        strategy: 補完戦略（カラム名: 補完方法）の辞書
                 補完方法: 'mean', 'median', 'mode', 'zero', 'none'のいずれか
        
    Returns:
        pd.DataFrame: 欠損値補完済みのDataFrame
    """
    df_filled = df.copy()
    
    for col, method in strategy.items():
        if col not in df_filled.columns:
            logger.warning(f"カラム {col} がDataFrameに存在しません")
            continue
        
        missing_count = df_filled[col].isna().sum()
        if missing_count == 0:
            logger.info(f"カラム {col} に欠損値はありません")
            continue
        
        if method == 'mean':
            if pd.api.types.is_numeric_dtype(df_filled[col]):
                df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
                logger.info(f"カラム {col} の欠損値 {missing_count} 件を平均値で補完しました")
            else:
                logger.warning(f"カラム {col} は数値型ではないため、平均値での補完はスキップします")
        
        elif method == 'median':
            if pd.api.types.is_numeric_dtype(df_filled[col]):
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())
                logger.info(f"カラム {col} の欠損値 {missing_count} 件を中央値で補完しました")
            else:
                logger.warning(f"カラム {col} は数値型ではないため、中央値での補完はスキップします")
        
        elif method == 'mode':
            mode_value = df_filled[col].mode()[0]
            df_filled[col] = df_filled[col].fillna(mode_value)
            logger.info(f"カラム {col} の欠損値 {missing_count} 件を最頻値で補完しました")
        
        elif method == 'zero':
            df_filled[col] = df_filled[col].fillna(0)
            logger.info(f"カラム {col} の欠損値 {missing_count} 件を0で補完しました")
        
        elif method == 'none':
            logger.info(f"カラム {col} の欠損値 {missing_count} 件は補完しません")
    
    return df_filled


if __name__ == "__main__":
    # モジュールのテスト
    try:
        from src.data.extraction import extract_race_base_info, extract_horse_race_results
        
        # テストデータの抽出
        logger.info("テストデータを抽出しています...")
        # 2020年の一部データでテスト
        test_race_df = extract_race_base_info('2020', '2020')
        test_horse_race_df = extract_horse_race_results('2020', '2020')
        
        # データクレンジングと型変換のテスト
        logger.info("レース基本情報のクレンジングと型変換をテストしています...")
        cleaned_race_df = clean_race_data(test_race_df)
        logger.info(f"クレンジング前: {test_race_df.shape}, クレンジング後: {cleaned_race_df.shape}")
        logger.info(f"追加されたカラム: {set(cleaned_race_df.columns) - set(test_race_df.columns)}")
        
        logger.info("出走馬情報のクレンジングと型変換をテストしています...")
        cleaned_horse_race_df = clean_horse_race_results(test_horse_race_df)
        logger.info(f"クレンジング前: {test_horse_race_df.shape}, クレンジング後: {cleaned_horse_race_df.shape}")
        logger.info(f"追加されたカラム: {set(cleaned_horse_race_df.columns) - set(test_horse_race_df.columns)}")
        
        # 外れ値処理と欠損値補完のテスト
        logger.info("外れ値処理をテストしています...")
        numeric_columns = cleaned_horse_race_df.select_dtypes(include=['number']).columns.tolist()
        processed_df = handle_outliers(cleaned_horse_race_df, numeric_columns)
        logger.info(f"処理前: {cleaned_horse_race_df.shape}, 処理後: {processed_df.shape}")
        
        logger.info("欠損値補完をテストしています...")
        missing_cols = processed_df.columns[processed_df.isna().any()].tolist()
        fill_strategy = {col: 'median' for col in missing_cols}
        filled_df = fill_missing_values(processed_df, fill_strategy)
        logger.info(f"補完前の欠損値数: {processed_df.isna().sum().sum()}, 補完後の欠損値数: {filled_df.isna().sum().sum()}")
        
    except Exception as e:
        logger.error(f"テスト中にエラーが発生しました: {e}")
