"""
データ抽出モジュール

JVDデータベースからのデータ抽出を行う関数を提供します。
主要テーブル（jvd_ra, jvd_se, jvd_hr, jvd_um, jvd_hn）からのデータ抽出をサポートします。
"""
import os
import logging
import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime

from src.data.database import execute_query, query_with_cache

# ロガーの設定
logger = logging.getLogger(__name__)

# チャンクサイズの取得
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '10000'))

# プロジェクトルートパスの取得
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def extract_race_base_info(
    start_year: str = '2000',
    end_year: str = '2024',
    cache: bool = True,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    レース基本情報（jvd_ra）を抽出
    
    Args:
        start_year: 開始年
        end_year: 終了年
        cache: キャッシュを使用するかどうか
        force_refresh: キャッシュを強制的に更新するかどうか
        
    Returns:
        pd.DataFrame: レース基本情報
    """
    query = """
    SELECT 
        kaisai_nen,
        kaisai_tsukihi,
        keibajo_code,
        race_bango,
        kaisai_nen || kaisai_tsukihi || keibajo_code || race_bango AS race_id,
        kyori,
        track_code,
        tenko_code,
        CASE 
            WHEN SUBSTRING(track_code, 1, 1) = '1' THEN babajotai_code_shiba
            ELSE babajotai_code_dirt
        END AS babajotai_code,
        grade_code,
        juryo_shubetsu_code,
        joken_code_1,
        joken_code_2,
        joken_code_3,
        joken_code_4,
        joken_code_5,
        shusso_tosu,
        race_syubetu_code
    FROM jvd_ra
    WHERE kaisai_nen BETWEEN :start_year AND :end_year
    ORDER BY kaisai_nen, kaisai_tsukihi, keibajo_code, race_bango
    """
    
    params = {'start_year': start_year, 'end_year': end_year}
    
    if cache:
        cache_name = f"race_base_info_{start_year}_{end_year}"
        return query_with_cache(query, cache_name, params, force_refresh)
    else:
        return execute_query(query, params)


def extract_horse_race_results(
    start_year: str = '2000',
    end_year: str = '2024',
    cache: bool = True,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    出走馬情報（jvd_se）と関連するレース情報を抽出
    
    Args:
        start_year: 開始年
        end_year: 終了年
        cache: キャッシュを使用するかどうか
        force_refresh: キャッシュを強制的に更新するかどうか
        
    Returns:
        pd.DataFrame: 出走馬情報とレース情報を結合したデータ
    """
    query = """
    SELECT 
        -- レース識別情報
        se.kaisai_nen,
        se.kaisai_tsukihi,
        se.keibajo_code,
        se.race_bango,
        se.kaisai_nen || se.kaisai_tsukihi || se.keibajo_code || se.race_bango AS race_id,
        
        -- 馬情報
        se.ketto_toroku_bango,
        TRIM(se.bamei) AS bamei,
        se.wakuban,
        se.umaban,
        se.barei,
        se.seibetsu_code,
        se.bataiju,
        se.zogen_fugo,
        se.zogen_sa,
        
        -- 騎手・調教師情報
        se.kishu_code,
        TRIM(se.kishumei_ryakusho) AS kishumei,
        se.chokyoshi_code,
        TRIM(se.chokyoshimei_ryakusho) AS chokyoshimei,
        
        -- レース結果
        se.kakutei_chakujun,
        se.soha_time,
        se.kohan_3f,
        
        -- オッズ情報
        se.tansho_odds,
        se.tansho_ninkijun,
        
        -- レース条件（jvd_raから）
        ra.kyori,
        ra.track_code,
        ra.tenko_code,
        CASE 
            WHEN SUBSTRING(ra.track_code, 1, 1) = '1' THEN ra.babajotai_code_shiba
            ELSE ra.babajotai_code_dirt
        END AS babajotai_code,
        ra.grade_code,
        ra.juryo_shubetsu_code,
        ra.shusso_tosu
    FROM jvd_se se
    JOIN jvd_ra ra ON se.kaisai_nen = ra.kaisai_nen
                AND se.kaisai_tsukihi = ra.kaisai_tsukihi
                AND se.keibajo_code = ra.keibajo_code
                AND se.race_bango = ra.race_bango
    WHERE se.kaisai_nen BETWEEN :start_year AND :end_year
    ORDER BY se.kaisai_nen, se.kaisai_tsukihi, se.keibajo_code, se.race_bango, se.umaban
    """
    
    params = {'start_year': start_year, 'end_year': end_year}
    
    if cache:
        cache_name = f"horse_race_results_{start_year}_{end_year}"
        return query_with_cache(query, cache_name, params, force_refresh)
    else:
        return execute_query(query, params)


def extract_horse_info(
    horse_ids: List[str] = None,
    cache: bool = True,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    馬基本情報（jvd_um）を抽出
    
    Args:
        horse_ids: 馬ID（血統登録番号）のリスト。Noneの場合は全馬を取得
        cache: キャッシュを使用するかどうか
        force_refresh: キャッシュを強制的に更新するかどうか
        
    Returns:
        pd.DataFrame: 馬基本情報
    """
    if horse_ids:
        query = """
        SELECT 
            ketto_toroku_bango,
            TRIM(bamei) AS bamei,
            seinengappi,
            seibetsu_code,
            ketto_joho_01a AS chichiuma_id,
            TRIM(ketto_joho_01b) AS chichiuma_name,
            ketto_joho_02a AS hahauma_id,
            TRIM(ketto_joho_02b) AS hahauma_name
        FROM jvd_um
        WHERE ketto_toroku_bango IN :horse_ids
        """
        params = {'horse_ids': tuple(horse_ids)}
    else:
        query = """
        SELECT 
            ketto_toroku_bango,
            TRIM(bamei) AS bamei,
            seinengappi,
            seibetsu_code,
            ketto_joho_01a AS chichiuma_id,
            TRIM(ketto_joho_01b) AS chichiuma_name,
            ketto_joho_02a AS hahauma_id,
            TRIM(ketto_joho_02b) AS hahauma_name
        FROM jvd_um
        """
        params = None
    
    if cache:
        if horse_ids:
            cache_name = f"horse_info_{len(horse_ids)}_horses"
        else:
            cache_name = "all_horse_info"
        return query_with_cache(query, cache_name, params, force_refresh)
    else:
        return execute_query(query, params)


def extract_pedigree_info(
    sire_ids: List[str] = None,
    cache: bool = True,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    血統・繁殖馬情報（jvd_hn）を抽出
    
    Args:
        sire_ids: 種牡馬ID（繁殖登録番号）のリスト。Noneの場合は全てを取得
        cache: キャッシュを使用するかどうか
        force_refresh: キャッシュを強制的に更新するかどうか
        
    Returns:
        pd.DataFrame: 血統・繁殖馬情報
    """
    if sire_ids:
        query = """
        SELECT 
            hanshoku_toroku_bango,
            ketto_toroku_bango,
            hatsunyusha_code,
            hatsunyushamei,
            hanshoku_kubun
        FROM jvd_hn
        WHERE hanshoku_toroku_bango IN :sire_ids
        """
        params = {'sire_ids': tuple(sire_ids)}
    else:
        query = """
        SELECT 
            hanshoku_toroku_bango,
            ketto_toroku_bango,
            hatsunyusha_code,
            hatsunyushamei,
            hanshoku_kubun
        FROM jvd_hn
        """
        params = None
    
    if cache:
        if sire_ids:
            cache_name = f"pedigree_info_{len(sire_ids)}_sires"
        else:
            cache_name = "all_pedigree_info"
        return query_with_cache(query, cache_name, params, force_refresh)
    else:
        return execute_query(query, params)


def extract_payouts(
    start_year: str = '2000',
    end_year: str = '2024',
    cache: bool = True,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    レース払戻情報（jvd_hr）を抽出
    
    Args:
        start_year: 開始年
        end_year: 終了年
        cache: キャッシュを使用するかどうか
        force_refresh: キャッシュを強制的に更新するかどうか
        
    Returns:
        pd.DataFrame: レース払戻情報
    """
    query = """
    SELECT 
        hr.kaisai_nen,
        hr.kaisai_tsukihi,
        hr.keibajo_code,
        hr.race_bango,
        hr.kaisai_nen || hr.kaisai_tsukihi || hr.keibajo_code || hr.race_bango AS race_id,
        
        -- 単勝
        hr.haraimodoshi_tansho_umaban1,
        hr.haraimodoshi_tansho_umaban2,
        hr.haraimodoshi_tansho_umaban3,
        hr.haraimodoshi_tansho_money1,
        hr.haraimodoshi_tansho_money2,
        hr.haraimodoshi_tansho_money3,
        
        -- 複勝
        hr.haraimodoshi_fukusho_umaban1,
        hr.haraimodoshi_fukusho_umaban2,
        hr.haraimodoshi_fukusho_umaban3,
        hr.haraimodoshi_fukusho_umaban4,
        hr.haraimodoshi_fukusho_umaban5,
        hr.haraimodoshi_fukusho_money1,
        hr.haraimodoshi_fukusho_money2,
        hr.haraimodoshi_fukusho_money3,
        hr.haraimodoshi_fukusho_money4,
        hr.haraimodoshi_fukusho_money5,
        
        -- 枠連
        hr.haraimodoshi_wakuren_kumiban1,
        hr.haraimodoshi_wakuren_kumiban2,
        hr.haraimodoshi_wakuren_kumiban3,
        hr.haraimodoshi_wakuren_money1,
        hr.haraimodoshi_wakuren_money2,
        hr.haraimodoshi_wakuren_money3,
        
        -- 馬連
        hr.haraimodoshi_umaren_kumiban1,
        hr.haraimodoshi_umaren_kumiban2,
        hr.haraimodoshi_umaren_kumiban3,
        hr.haraimodoshi_umaren_money1,
        hr.haraimodoshi_umaren_money2,
        hr.haraimodoshi_umaren_money3,
        
        -- ワイド
        hr.haraimodoshi_wide_kumiban1,
        hr.haraimodoshi_wide_kumiban2,
        hr.haraimodoshi_wide_kumiban3,
        hr.haraimodoshi_wide_kumiban4,
        hr.haraimodoshi_wide_kumiban5,
        hr.haraimodoshi_wide_kumiban6,
        hr.haraimodoshi_wide_kumiban7,
        hr.haraimodoshi_wide_money1,
        hr.haraimodoshi_wide_money2,
        hr.haraimodoshi_wide_money3,
        hr.haraimodoshi_wide_money4,
        hr.haraimodoshi_wide_money5,
        hr.haraimodoshi_wide_money6,
        hr.haraimodoshi_wide_money7,
        
        -- 馬単
        hr.haraimodoshi_umatan_kumiban1,
        hr.haraimodoshi_umatan_kumiban2,
        hr.haraimodoshi_umatan_kumiban3,
        hr.haraimodoshi_umatan_kumiban4,
        hr.haraimodoshi_umatan_kumiban5,
        hr.haraimodoshi_umatan_kumiban6,
        hr.haraimodoshi_umatan_money1,
        hr.haraimodoshi_umatan_money2,
        hr.haraimodoshi_umatan_money3,
        hr.haraimodoshi_umatan_money4,
        hr.haraimodoshi_umatan_money5,
        hr.haraimodoshi_umatan_money6,
        
        -- 三連複
        hr.haraimodoshi_sanrenpuku_kumiban1,
        hr.haraimodoshi_sanrenpuku_kumiban2,
        hr.haraimodoshi_sanrenpuku_kumiban3,
        hr.haraimodoshi_sanrenpuku_money1,
        hr.haraimodoshi_sanrenpuku_money2,
        hr.haraimodoshi_sanrenpuku_money3,
        
        -- 三連単
        hr.haraimodoshi_sanrentan_kumiban1,
        hr.haraimodoshi_sanrentan_kumiban2,
        hr.haraimodoshi_sanrentan_kumiban3,
        hr.haraimodoshi_sanrentan_kumiban4,
        hr.haraimodoshi_sanrentan_kumiban5,
        hr.haraimodoshi_sanrentan_kumiban6,
        hr.haraimodoshi_sanrentan_money1,
        hr.haraimodoshi_sanrentan_money2,
        hr.haraimodoshi_sanrentan_money3,
        hr.haraimodoshi_sanrentan_money4,
        hr.haraimodoshi_sanrentan_money5,
        hr.haraimodoshi_sanrentan_money6
    FROM jvd_hr hr
    WHERE hr.kaisai_nen BETWEEN :start_year AND :end_year
    """
    
    params = {'start_year': start_year, 'end_year': end_year}
    
    if cache:
        cache_name = f"race_payouts_{start_year}_{end_year}"
        return query_with_cache(query, cache_name, params, force_refresh)
    else:
        return execute_query(query, params)


def extract_merged_base_data(
    start_year: str = '2000',
    end_year: str = '2024',
    cache: bool = True,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    基本データを結合して取得（レース情報・出走馬情報・血統情報）
    
    Args:
        start_year: 開始年
        end_year: 終了年
        cache: キャッシュを使用するかどうか
        force_refresh: キャッシュを強制的に更新するかどうか
        
    Returns:
        pd.DataFrame: 結合された基本データ
    """
    query = """
    WITH race_horse_data AS (
        SELECT 
            -- レース識別情報
            se.kaisai_nen,
            se.kaisai_tsukihi,
            se.keibajo_code,
            se.race_bango,
            se.kaisai_nen || se.kaisai_tsukihi || se.keibajo_code || se.race_bango AS race_id,
            
            -- 馬情報
            se.ketto_toroku_bango,
            TRIM(se.bamei) AS bamei,
            se.wakuban,
            se.umaban,
            se.barei,
            se.seibetsu_code,
            se.bataiju,
            se.zogen_fugo,
            se.zogen_sa,
            
            -- 騎手・調教師情報
            se.kishu_code,
            TRIM(se.kishumei_ryakusho) AS kishumei,
            se.chokyoshi_code,
            TRIM(se.chokyoshimei_ryakusho) AS chokyoshimei,
            
            -- レース結果
            se.kakutei_chakujun,
            se.soha_time,
            se.kohan_3f,
            
            -- オッズ情報
            se.tansho_odds,
            se.tansho_ninkijun,
            
            -- レース条件（jvd_raから）
            ra.kyori,
            ra.track_code,
            ra.tenko_code,
            CASE 
                WHEN SUBSTRING(ra.track_code, 1, 1) = '1' THEN ra.babajotai_code_shiba
                ELSE ra.babajotai_code_dirt
            END AS babajotai_code,
            ra.grade_code,
            ra.juryo_shubetsu_code,
            ra.shusso_tosu
        FROM jvd_se se
        JOIN jvd_ra ra ON se.kaisai_nen = ra.kaisai_nen
                    AND se.kaisai_tsukihi = ra.kaisai_tsukihi
                    AND se.keibajo_code = ra.keibajo_code
                    AND se.race_bango = ra.race_bango
        WHERE se.kaisai_nen BETWEEN :start_year AND :end_year
    )
    SELECT 
        rhd.*,
        
        -- 馬の基本情報（jvd_umから）
        um.seinengappi,
        
        -- 血統情報
        um.ketto_joho_01a AS chichiuma_id,
        TRIM(um.ketto_joho_01b) AS chichiuma_name,
        um.ketto_joho_02a AS hahauma_id,
        TRIM(um.ketto_joho_02b) AS hahauma_name,
        um.ketto_joho_03a AS sofu1_id,
        TRIM(um.ketto_joho_03b) AS sofu1_name,
        um.ketto_joho_04a AS sobo1_id,
        TRIM(um.ketto_joho_04b) AS sobo1_name,
        um.ketto_joho_05a AS sofu2_id, 
        TRIM(um.ketto_joho_05b) AS sofu2_name,
        um.ketto_joho_06a AS sobo2_id,
        TRIM(um.ketto_joho_06b) AS sobo2_name
    FROM race_horse_data rhd
    LEFT JOIN jvd_um um ON rhd.ketto_toroku_bango = um.ketto_toroku_bango
    ORDER BY rhd.kaisai_nen, rhd.kaisai_tsukihi, rhd.keibajo_code, rhd.race_bango, rhd.umaban
    """
    
    params = {'start_year': start_year, 'end_year': end_year}
    
    if cache:
        cache_name = f"merged_base_data_{start_year}_{end_year}"
        return query_with_cache(query, cache_name, params, force_refresh)
    else:
        return execute_query(query, params)


def extract_data_by_year(
    extract_func, 
    start_year: str = '2000',
    end_year: str = '2024',
    year_chunk_size: int = 5,
    save_path: Optional[str] = None,
    force_refresh: bool = False
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    年度別にデータを抽出し、キャッシュ・結合する
    
    Args:
        extract_func: 抽出関数
        start_year: 開始年
        end_year: 終了年
        year_chunk_size: 一度に抽出する年数
        save_path: 保存先パス
        force_refresh: キャッシュを強制的に更新するかどうか
        
    Returns:
        pd.DataFrame または DataFrame のリスト
    """
    # 年度リストを作成
    years = [str(y) for y in range(int(start_year), int(end_year) + 1)]
    
    # year_chunk_size 年ごとにデータを抽出
    chunks = []
    for i in range(0, len(years), year_chunk_size):
        chunk_start_year = years[i]
        chunk_end_year = years[min(i + year_chunk_size - 1, len(years) - 1)]
        
        logger.info(f"{chunk_start_year}年から{chunk_end_year}年のデータを抽出しています...")
        chunk_df = extract_func(chunk_start_year, chunk_end_year, True, force_refresh)
        chunks.append(chunk_df)
        logger.info(f"{len(chunk_df)}行のデータを抽出しました")
    
    # 全てのチャンクを結合
    combined_df = pd.concat(chunks, ignore_index=True)
    logger.info(f"合計{len(combined_df)}行のデータを抽出しました")
    
    # 結果を保存
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_pickle(save_path)
        logger.info(f"結果を {save_path} に保存しました")
    
    return combined_df


if __name__ == "__main__":
    # モジュールのテスト
    try:
        # 開始年と終了年
        start_year = '2020'
        end_year = '2020'
        
        # レース基本情報の抽出
        logger.info(f"{start_year}年から{end_year}年のレース基本情報を抽出しています...")
        race_df = extract_race_base_info(start_year, end_year)
        logger.info(f"{len(race_df)}行のレースデータを抽出しました")
        
        # 出走馬情報の抽出
        logger.info(f"{start_year}年から{end_year}年の出走馬情報を抽出しています...")
        horse_race_df = extract_horse_race_results(start_year, end_year)
        logger.info(f"{len(horse_race_df)}行の出走馬データを抽出しました")
        
        # 血統情報のサンプル抽出
        logger.info("血統情報のサンプルを抽出しています...")
        horse_sample = horse_race_df['ketto_toroku_bango'].head(10).tolist()
        horse_info_df = extract_horse_info(horse_sample)
        logger.info(f"{len(horse_info_df)}行の馬情報を抽出しました")
        
    except Exception as e:
        logger.error(f"テスト中にエラーが発生しました: {e}")
