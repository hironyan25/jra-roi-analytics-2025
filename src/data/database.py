"""
データベース接続モジュール

JVDデータベースへの接続とクエリ実行を行うユーティリティ関数を提供します。
"""
import os
import logging
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union, List, Any
from sqlalchemy import create_engine, text, Table, MetaData
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

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

# プロジェクトルートパスの取得
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_db_config(environment: str = None) -> Dict[str, str]:
    """
    データベース設定をyamlファイルから読み込む
    
    Args:
        environment: 環境名（development, test, production）
        
    Returns:
        Dict[str, str]: データベース接続設定を含む辞書
    """
    if environment is None:
        environment = os.getenv('ENVIRONMENT', 'development')
    
    config_path = PROJECT_ROOT / 'config' / 'database.yml'
    
    # 設定ファイルが存在しない場合はデフォルト値を使用
    if not config_path.exists():
        logger.warning(f"データベース設定ファイルが見つかりません: {config_path}")
        return {
            'host': os.getenv('DB_HOST', '127.0.0.1'),
            'port': os.getenv('DB_PORT', '5432'),
            'dbname': os.getenv('DB_NAME', 'pckeiba'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASS', 'postgres')
        }
    
    # yamlから設定を読み込む
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 環境に合わせた設定を返す
    return config.get(environment, config.get('default', {}))


def get_engine(config: Dict[str, str] = None) -> Engine:
    """
    SQLAlchemy engineを取得
    
    Args:
        config: データベース接続設定。省略時は設定ファイルから読み込み
        
    Returns:
        Engine: SQLAlchemy engine
    """
    if config is None:
        config = load_db_config()
    
    # 設定内の環境変数プレースホルダーを解決
    for key, value in config.items():
        if isinstance(value, str) and value.startswith('<%=') and value.endswith('%>'):
            env_var = value.strip('<%= %>').strip()
            config[key] = os.getenv(env_var, '')
    
    # 接続文字列の作成
    conn_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['dbname']}"
    
    return create_engine(conn_string)


def execute_query(query: str, params: Dict = None, chunksize: int = None) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    SQLクエリを実行し、結果をPandas DataFrameで返す
    
    Args:
        query: 実行するSQLクエリ
        params: クエリパラメータ
        chunksize: 結果をチャンクで取得する場合のサイズ
        
    Returns:
        pd.DataFrame または DataFrame のリスト
    """
    engine = get_engine()
    try:
        if params:
            return pd.read_sql_query(text(query), engine, params=params, chunksize=chunksize)
        return pd.read_sql_query(text(query), engine, chunksize=chunksize)
    except SQLAlchemyError as e:
        logger.error(f"クエリ実行中にエラーが発生しました: {e}")
        raise


def query_with_cache(
    query: str, 
    cache_name: str, 
    params: Dict = None, 
    force_refresh: bool = False,
    cache_dir: str = None
) -> pd.DataFrame:
    """
    キャッシュ機能付きクエリ実行
    
    Args:
        query: 実行するSQLクエリ
        cache_name: キャッシュファイル名（拡張子なし）
        params: クエリパラメータ
        force_refresh: キャッシュを無視して強制的に再取得するかどうか
        cache_dir: キャッシュディレクトリ
        
    Returns:
        pd.DataFrame: クエリ結果
    """
    if cache_dir is None:
        cache_dir = os.getenv('CACHE_DIR', './data/cache')
    
    cache_path = Path(cache_dir) / f"{cache_name}.pkl"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    if not force_refresh and cache_path.exists():
        logger.info(f"キャッシュから{cache_name}を読み込んでいます...")
        return pd.read_pickle(cache_path)
    
    logger.info(f"データベースから{cache_name}を取得しています...")
    df = execute_query(query, params)
    
    # 結果をキャッシュに保存
    df.to_pickle(cache_path)
    logger.info(f"{cache_name}をキャッシュに保存しました: {len(df)}行")
    
    return df


def get_table_info():
    """
    データベース内のテーブル情報を取得
    
    Returns:
        Dict: テーブル名をキー、カラム情報をバリューとする辞書
    """
    engine = get_engine()
    meta = MetaData()
    meta.reflect(bind=engine)
    
    tables = {}
    for table_name, table in meta.tables.items():
        columns = {}
        for column in table.columns:
            columns[column.name] = {
                'type': str(column.type),
                'primary_key': column.primary_key,
                'nullable': column.nullable
            }
        tables[table_name] = columns
    
    return tables


def execute_raw_sql(query: str, params: Dict = None) -> Any:
    """
    SQLクエリを直接実行
    
    Args:
        query: 実行するSQLクエリ
        params: クエリパラメータ
        
    Returns:
        クエリの結果
    """
    engine = get_engine()
    try:
        with engine.connect() as connection:
            if params:
                result = connection.execute(text(query), params)
            else:
                result = connection.execute(text(query))
            return result
    except SQLAlchemyError as e:
        logger.error(f"クエリ実行中にエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    # モジュールのテスト
    try:
        engine = get_engine()
        logger.info("データベースに接続しました")
        
        # テーブル情報の取得
        table_info = get_table_info()
        logger.info(f"テーブル数: {len(table_info)}")
        for table_name, columns in table_info.items():
            logger.info(f"テーブル: {table_name}, カラム数: {len(columns)}")
        
        # サンプルクエリの実行
        query = "SELECT 1 as test"
        result = execute_query(query)
        logger.info(f"テストクエリ実行結果: {result}")
        
    except Exception as e:
        logger.error(f"テスト中にエラーが発生しました: {e}")
