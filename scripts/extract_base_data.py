#!/usr/bin/env python
"""
基本データ抽出と前処理スクリプト

JVDデータベースから基本データを抽出し、前処理を行います。
"""
import os
import sys
import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# プロジェクトルートをシステムパスに追加
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# 自作モジュールのインポート
from src.data.extraction import (
    extract_race_base_info,
    extract_horse_race_results,
    extract_horse_info,
    extract_pedigree_info,
    extract_payouts,
    extract_merged_base_data,
    extract_data_by_year
)
from src.data.preprocessing import (
    clean_race_data,
    clean_horse_race_results,
    clean_horse_info,
    clean_pedigree_info,
    clean_payouts,
    clean_merged_base_data,
    handle_outliers,
    fill_missing_values
)

# 環境変数の読み込み
load_dotenv()

# ロガーの設定
logger = logging.getLogger(__name__)
log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 出力ディレクトリの設定
DATA_DIR = project_root / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_DIR = DATA_DIR / "feature_store"

# 各ディレクトリの作成
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def extract_raw_data(
    start_year: str,
    end_year: str,
    extract_types: list,
    force_refresh: bool = False
) -> dict:
    """
    生データを抽出してRAW_DIRに保存
    
    Args:
        start_year: 開始年
        end_year: 終了年
        extract_types: 抽出するデータタイプのリスト
                      ['race', 'horse_race', 'horse', 'pedigree', 'payout', 'merged']
        force_refresh: キャッシュを強制的に更新するかどうか
    
    Returns:
        dict: 抽出したデータのDataFrameを格納した辞書
    """
    extracted_data = {}
    
    # レース基本情報の抽出
    if 'race' in extract_types:
        logger.info(f"{start_year}年から{end_year}年のレース基本情報を抽出しています...")
        race_df = extract_data_by_year(
            extract_race_base_info,
            start_year,
            end_year,
            year_chunk_size=5,
            save_path=RAW_DIR / f"race_base_info_{start_year}_{end_year}.pkl",
            force_refresh=force_refresh
        )
        extracted_data['race'] = race_df
        logger.info(f"{len(race_df)}行のレースデータを抽出しました")
    
    # 出走馬情報の抽出
    if 'horse_race' in extract_types:
        logger.info(f"{start_year}年から{end_year}年の出走馬情報を抽出しています...")
        horse_race_df = extract_data_by_year(
            extract_horse_race_results,
            start_year,
            end_year,
            year_chunk_size=3,
            save_path=RAW_DIR / f"horse_race_results_{start_year}_{end_year}.pkl",
            force_refresh=force_refresh
        )
        extracted_data['horse_race'] = horse_race_df
        logger.info(f"{len(horse_race_df)}行の出走馬データを抽出しました")
    
    # 馬基本情報の抽出
    if 'horse' in extract_types:
        if 'horse_race' in extract_types and 'horse_race' in extracted_data:
            # 出走馬データから馬IDを取得
            horse_ids = extracted_data['horse_race']['ketto_toroku_bango'].unique().tolist()
        else:
            logger.warning("馬IDを取得するための出走馬データがないため、全馬情報を抽出します")
            horse_ids = None
        
        logger.info("馬基本情報を抽出しています...")
        horse_info_df = extract_horse_info(
            horse_ids=horse_ids,
            cache=True,
            force_refresh=force_refresh
        )
        horse_info_df.to_pickle(RAW_DIR / f"horse_info.pkl")
        extracted_data['horse'] = horse_info_df
        logger.info(f"{len(horse_info_df)}行の馬情報を抽出しました")
    
    # 血統情報の抽出
    if 'pedigree' in extract_types:
        if 'horse' in extract_types and 'horse' in extracted_data:
            # 馬基本情報から種牡馬IDを取得
            sire_ids = extracted_data['horse']['chichiuma_id'].unique().tolist()
            sire_ids = [sid for sid in sire_ids if sid and str(sid) != '0000000000']
        else:
            logger.warning("種牡馬IDを取得するための馬基本情報がないため、全血統情報を抽出します")
            sire_ids = None
        
        logger.info("血統情報を抽出しています...")
        pedigree_df = extract_pedigree_info(
            sire_ids=sire_ids,
            cache=True,
            force_refresh=force_refresh
        )
        pedigree_df.to_pickle(RAW_DIR / f"pedigree_info.pkl")
        extracted_data['pedigree'] = pedigree_df
        logger.info(f"{len(pedigree_df)}行の血統情報を抽出しました")
    
    # 払戻情報の抽出
    if 'payout' in extract_types:
        logger.info(f"{start_year}年から{end_year}年の払戻情報を抽出しています...")
        payout_df = extract_data_by_year(
            extract_payouts,
            start_year,
            end_year,
            year_chunk_size=5,
            save_path=RAW_DIR / f"race_payouts_{start_year}_{end_year}.pkl",
            force_refresh=force_refresh
        )
        extracted_data['payout'] = payout_df
        logger.info(f"{len(payout_df)}行の払戻データを抽出しました")
    
    # マージされた基本データの抽出
    if 'merged' in extract_types:
        logger.info(f"{start_year}年から{end_year}年の結合基本データを抽出しています...")
        merged_df = extract_data_by_year(
            extract_merged_base_data,
            start_year,
            end_year,
            year_chunk_size=2,
            save_path=RAW_DIR / f"merged_base_data_{start_year}_{end_year}.pkl",
            force_refresh=force_refresh
        )
        extracted_data['merged'] = merged_df
        logger.info(f"{len(merged_df)}行の結合基本データを抽出しました")
    
    return extracted_data


def preprocess_data(
    extracted_data: dict,
    start_year: str,
    end_year: str
) -> dict:
    """
    抽出したデータの前処理を行いPROCESSED_DIRに保存
    
    Args:
        extracted_data: 抽出したデータのDataFrameを格納した辞書
        start_year: 開始年
        end_year: 終了年
    
    Returns:
        dict: 前処理済みデータのDataFrameを格納した辞書
    """
    processed_data = {}
    
    # レース基本情報の前処理
    if 'race' in extracted_data:
        logger.info("レース基本情報を前処理しています...")
        race_df = extracted_data['race']
        processed_race_df = clean_race_data(race_df)
        
        # 外れ値の処理
        numeric_cols = processed_race_df.select_dtypes(include=['number']).columns.tolist()
        processed_race_df = handle_outliers(processed_race_df, numeric_cols)
        
        # 欠損値の補完
        missing_cols = processed_race_df.columns[processed_race_df.isna().any()].tolist()
        fill_strategy = {col: 'median' if pd.api.types.is_numeric_dtype(processed_race_df[col]) else 'mode' for col in missing_cols}
        processed_race_df = fill_missing_values(processed_race_df, fill_strategy)
        
        # 処理済みデータの保存
        processed_race_df.to_pickle(PROCESSED_DIR / f"race_base_info_{start_year}_{end_year}_processed.pkl")
        processed_data['race'] = processed_race_df
        logger.info(f"レース基本情報の前処理が完了しました: {len(processed_race_df)}行")
    
    # 出走馬情報の前処理
    if 'horse_race' in extracted_data:
        logger.info("出走馬情報を前処理しています...")
        horse_race_df = extracted_data['horse_race']
        processed_horse_race_df = clean_horse_race_results(horse_race_df)
        
        # 外れ値の処理
        numeric_cols = processed_horse_race_df.select_dtypes(include=['number']).columns.tolist()
        processed_horse_race_df = handle_outliers(processed_horse_race_df, numeric_cols)
        
        # 欠損値の補完
        missing_cols = processed_horse_race_df.columns[processed_horse_race_df.isna().any()].tolist()
        fill_strategy = {col: 'median' if pd.api.types.is_numeric_dtype(processed_horse_race_df[col]) else 'mode' for col in missing_cols}
        processed_horse_race_df = fill_missing_values(processed_horse_race_df, fill_strategy)
        
        # 処理済みデータの保存
        processed_horse_race_df.to_pickle(PROCESSED_DIR / f"horse_race_results_{start_year}_{end_year}_processed.pkl")
        processed_data['horse_race'] = processed_horse_race_df
        logger.info(f"出走馬情報の前処理が完了しました: {len(processed_horse_race_df)}行")
    
    # 馬基本情報の前処理
    if 'horse' in extracted_data:
        logger.info("馬基本情報を前処理しています...")
        horse_df = extracted_data['horse']
        processed_horse_df = clean_horse_info(horse_df)
        
        # 処理済みデータの保存
        processed_horse_df.to_pickle(PROCESSED_DIR / f"horse_info_processed.pkl")
        processed_data['horse'] = processed_horse_df
        logger.info(f"馬基本情報の前処理が完了しました: {len(processed_horse_df)}行")
    
    # 血統情報の前処理
    if 'pedigree' in extracted_data:
        logger.info("血統情報を前処理しています...")
        pedigree_df = extracted_data['pedigree']
        processed_pedigree_df = clean_pedigree_info(pedigree_df)
        
        # 処理済みデータの保存
        processed_pedigree_df.to_pickle(PROCESSED_DIR / f"pedigree_info_processed.pkl")
        processed_data['pedigree'] = processed_pedigree_df
        logger.info(f"血統情報の前処理が完了しました: {len(processed_pedigree_df)}行")
    
    # 払戻情報の前処理
    if 'payout' in extracted_data:
        logger.info("払戻情報を前処理しています...")
        payout_df = extracted_data['payout']
        processed_payout_df = clean_payouts(payout_df)
        
        # 処理済みデータの保存
        processed_payout_df.to_pickle(PROCESSED_DIR / f"race_payouts_{start_year}_{end_year}_processed.pkl")
        processed_data['payout'] = processed_payout_df
        logger.info(f"払戻情報の前処理が完了しました: {len(processed_payout_df)}行")
    
    # マージされた基本データの前処理
    if 'merged' in extracted_data:
        logger.info("結合基本データを前処理しています...")
        merged_df = extracted_data['merged']
        processed_merged_df = clean_merged_base_data(merged_df)
        
        # 外れ値の処理
        numeric_cols = processed_merged_df.select_dtypes(include=['number']).columns.tolist()
        processed_merged_df = handle_outliers(processed_merged_df, numeric_cols)
        
        # 欠損値の補完
        missing_cols = processed_merged_df.columns[processed_merged_df.isna().any()].tolist()
        fill_strategy = {col: 'median' if pd.api.types.is_numeric_dtype(processed_merged_df[col]) else 'mode' for col in missing_cols}
        processed_merged_df = fill_missing_values(processed_merged_df, fill_strategy)
        
        # 処理済みデータの保存
        processed_merged_df.to_pickle(PROCESSED_DIR / f"merged_base_data_{start_year}_{end_year}_processed.pkl")
        processed_data['merged'] = processed_merged_df
        logger.info(f"結合基本データの前処理が完了しました: {len(processed_merged_df)}行")
    
    return processed_data


def split_train_test(
    df: pd.DataFrame,
    train_years: list,
    validation_years: list,
    test_years: list,
    time_col: str = 'race_date'
) -> tuple:
    """
    データを訓練・検証・テストセットに分割
    
    Args:
        df: 分割するDataFrame
        train_years: 訓練データの年（リスト）
        validation_years: 検証データの年（リスト）
        test_years: テストデータの年（リスト）
        time_col: 時間カラム
    
    Returns:
        tuple: (train_df, validation_df, test_df)
    """
    # 文字列の場合は日付型に変換
    if df[time_col].dtype == 'object':
        df[time_col] = pd.to_datetime(df[time_col])
    
    # 年を抽出
    df['year'] = df[time_col].dt.year
    
    # データを分割
    train_df = df[df['year'].isin(train_years)].copy()
    validation_df = df[df['year'].isin(validation_years)].copy()
    test_df = df[df['year'].isin(test_years)].copy()
    
    logger.info(f"データ分割: 訓練{len(train_df)}行, 検証{len(validation_df)}行, テスト{len(test_df)}行")
    
    # 一時列を削除
    if 'year' not in df.columns:
        train_df.drop(columns=['year'], inplace=True)
        validation_df.drop(columns=['year'], inplace=True)
        test_df.drop(columns=['year'], inplace=True)
    
    return train_df, validation_df, test_df


def save_split_data(
    df: pd.DataFrame,
    filename_base: str,
    train_years: list,
    validation_years: list,
    test_years: list,
    time_col: str = 'race_date'
):
    """
    データを訓練・検証・テストセットに分割して保存
    
    Args:
        df: 分割するDataFrame
        filename_base: 保存ファイル名の基本部分
        train_years: 訓練データの年（リスト）
        validation_years: 検証データの年（リスト）
        test_years: テストデータの年（リスト）
        time_col: 時間カラム
    """
    train_df, validation_df, test_df = split_train_test(
        df, train_years, validation_years, test_years, time_col
    )
    
    # データを保存
    train_df.to_pickle(PROCESSED_DIR / f"{filename_base}_train.pkl")
    validation_df.to_pickle(PROCESSED_DIR / f"{filename_base}_validation.pkl")
    test_df.to_pickle(PROCESSED_DIR / f"{filename_base}_test.pkl")
    
    logger.info(f"{filename_base}の訓練・検証・テストデータを保存しました")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='JRAデータの抽出と前処理を行うスクリプト')
    parser.add_argument('--start_year', type=str, default='2000', help='抽出開始年')
    parser.add_argument('--end_year', type=str, default='2024', help='抽出終了年')
    parser.add_argument('--extract_types', type=str, default='race,horse_race,merged', 
                        help='抽出するデータタイプ (カンマ区切り): race,horse_race,horse,pedigree,payout,merged')
    parser.add_argument('--force_refresh', action='store_true', help='キャッシュを強制的に更新')
    parser.add_argument('--split', action='store_true', help='訓練・検証・テストデータに分割するかどうか')
    args = parser.parse_args()
    
    # データタイプをリストに変換
    extract_types = args.extract_types.split(',')
    
    # 年度の範囲を配列に変換
    years = list(range(int(args.start_year), int(args.end_year) + 1))
    
    # データの抽出
    logger.info(f"{args.start_year}年から{args.end_year}年のデータを抽出します")
    extracted_data = extract_raw_data(
        args.start_year,
        args.end_year,
        extract_types,
        args.force_refresh
    )
    
    # データの前処理
    logger.info("抽出したデータの前処理を行います")
    processed_data = preprocess_data(
        extracted_data,
        args.start_year,
        args.end_year
    )
    
    # 訓練・検証・テストデータに分割
    if args.split:
        # 年度を訓練・検証・テストに分割
        # 例: 2000-2020を訓練、2021-2022を検証、2023-2024をテスト
        train_end_year = min(2020, int(args.end_year))
        validation_end_year = min(2022, int(args.end_year))
        
        train_years = list(range(int(args.start_year), train_end_year + 1))
        validation_years = list(range(train_end_year + 1, validation_end_year + 1))
        test_years = list(range(validation_end_year + 1, int(args.end_year) + 1))
        
        logger.info(f"データを分割します: 訓練{train_years}, 検証{validation_years}, テスト{test_years}")
        
        # レース基本情報の分割
        if 'race' in processed_data:
            save_split_data(
                processed_data['race'],
                'race_base_info',
                train_years,
                validation_years,
                test_years
            )
        
        # 出走馬情報の分割
        if 'horse_race' in processed_data:
            save_split_data(
                processed_data['horse_race'],
                'horse_race_results',
                train_years,
                validation_years,
                test_years
            )
        
        # マージされた基本データの分割
        if 'merged' in processed_data:
            save_split_data(
                processed_data['merged'],
                'merged_base_data',
                train_years,
                validation_years,
                test_years
            )
    
    logger.info("全ての処理が完了しました")


if __name__ == "__main__":
    main()
