import sys
import json
import numpy as np
import argparse
import multiprocessing as mp
import time
import math
import psycopg2
import pymysql
import sqlite3
from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_file_logging(output_log_path):
    """Set up file logging handler"""
    # Ensure output directory exists
    output_dir = os.path.dirname(output_log_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file in the same directory
    log_filename = os.path.join(output_dir, f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Add file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

# Global variable for parallel execution results
exec_result = []
total_queries = 0
processed_queries = 0

def result_callback(result):
    global processed_queries
    exec_result.append(result)
    processed_queries += 1
    logger.debug(f"Processed query {processed_queries}/{total_queries}")

# ================ Utility Functions ================
def load_jsonl(file_path):
    """Load data from a JSONL file"""
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    return data

def load_json(dir):
    """Load data from a JSON file"""
    with open(dir, "r") as j:
        contents = json.loads(j.read())
    return contents

def connect_postgresql():
    """Connect to PostgreSQL database"""
    db = psycopg2.connect(
        "dbname=bird user=postgres host=localhost password=li123911 port=5432"
    )
    return db

def connect_mysql():
    """Connect to MySQL database"""
    db = pymysql.connect(
        host="localhost",
        user="root",
        password="li123911",
        database="BIRD",
        unix_socket="/var/run/mysqld/mysqld.sock"
    )
    return db

def connect_db(sql_dialect, db_path):
    """Connect to database based on SQL dialect"""
    if sql_dialect == "SQLite":
        conn = sqlite3.connect(db_path)
    elif sql_dialect == "MySQL":
        conn = connect_mysql()
    elif sql_dialect == "PostgreSQL":
        conn = connect_postgresql()
    else:
        raise ValueError("Unsupported SQL dialect")
    return conn

def execute_sql(sql, db_path, sql_dialect, calculate_func=None, return_time=False):
    """Execute SQL query and optionally calculate score or return execution time"""
    conn = connect_db(sql_dialect, db_path)
    cursor = conn.cursor()
    start_time = time.time()
    cursor.execute(sql)
    res = cursor.fetchall()
    conn.close()
    
    if return_time:
        return time.time() - start_time
    elif calculate_func:
        return calculate_func(res)
    return res

def package_sqls(sql_path, db_root_path, mode="pred"):
    """Package SQL queries and their corresponding database paths"""
    clean_sqls = []
    db_path_list = []
    if mode == "pred":
        sql_data = json.load(open(sql_path, "r"))
        for _, sql_str in sql_data.items():
            if isinstance(sql_str, str):
                try:
                    sql, db_name = sql_str.split("\t----- bird -----\t")
                except ValueError:
                    sql = sql_str.strip()
                    db_name = "financial"
            else:
                sql = " "
                db_name = "financial"               
            clean_sqls.append(sql)
    elif mode == "gt":
        sqls = open(sql_path)
        sql_txt = sqls.readlines()
        for idx, sql_str in enumerate(sql_txt):
            sql, db_name = sql_str.strip().split("\t")
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + "/" + db_name + ".sqlite")
    return clean_sqls, db_path_list

def sort_results(list_of_dicts):
    """Sort results by SQL index"""
    return sorted(list_of_dicts, key=lambda x: x["sql_idx"])

def print_data(score_lists, count_lists, metric="F1 Score", result_log_file=None):
    """Print and log evaluation results"""
    levels = ["simple", "moderate", "challenging", "total"]
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format("count", *count_lists))
    print(f"======================================    {metric}    =====================================")
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format(metric, *score_lists))
    
    if result_log_file is not None:
        with open(result_log_file, "a") as log_file:
            log_file.write(f"start calculate {metric}\n")
            log_file.write("{:20} {:20} {:20} {:20} {:20}\n".format("", *levels))
            log_file.write("{:20} {:<20} {:<20} {:<20} {:<20}\n".format("count", *count_lists))
            log_file.write(f"======================================    {metric}   =====================================\n")
            log_file.write("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}\n".format(metric, *score_lists))
            log_file.write("===========================================================================================\n")
            log_file.write(f"Finished {metric} evaluation for mini dev set\n")
            log_file.write("\n")

# ================ EX Metric ================
def calculate_ex(predicted_res, ground_truth_res):
    """Calculate exact match score (1 if sets match exactly, 0 otherwise)"""
    return 1 if set(predicted_res) == set(ground_truth_res) else 0

# ================ F1 Metric ================
def calculate_row_match(predicted_row, ground_truth_row):
    """Calculate the matching percentage for a single row"""
    total_columns = len(ground_truth_row)
    matches = 0
    element_in_pred_only = 0
    element_in_truth_only = 0
    
    for pred_val in predicted_row:
        if pred_val in ground_truth_row:
            matches += 1
        else:
            element_in_pred_only += 1
            
    for truth_val in ground_truth_row:
        if truth_val not in predicted_row:
            element_in_truth_only += 1
            
    match_percentage = matches / total_columns
    pred_only_percentage = element_in_pred_only / total_columns
    truth_only_percentage = element_in_truth_only / total_columns
    return match_percentage, pred_only_percentage, truth_only_percentage

def calculate_f1_score(predicted, ground_truth):
    """Calculate F1 score based on predicted and ground truth results"""
    if not predicted and not ground_truth:
        return 1.0

    predicted_set = set(predicted) if predicted else set()
    ground_truth_set = set(ground_truth)
    predicted = list(predicted_set)
    ground_truth = list(ground_truth_set)

    match_scores = []
    pred_only_scores = []
    truth_only_scores = []
    
    for i, gt_row in enumerate(ground_truth):
        if i >= len(predicted):
            match_scores.append(0)
            truth_only_scores.append(1)
            continue
        pred_row = predicted[i]
        match_score, pred_only_score, truth_only_score = calculate_row_match(pred_row, gt_row)
        match_scores.append(match_score)
        pred_only_scores.append(pred_only_score)
        truth_only_scores.append(truth_only_score)

    for i in range(len(predicted) - len(ground_truth)):
        match_scores.append(0)
        pred_only_scores.append(1)
        truth_only_scores.append(0)

    tp = sum(match_scores)
    fp = sum(pred_only_scores)
    fn = sum(truth_only_scores)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1_score

# ================ VES Metric ================
def clean_abnormal(input):
    """Remove outliers from execution time measurements"""
    input = np.asarray(input)
    processed_list = []
    mean = np.mean(input, axis=0)
    std = np.std(input, axis=0)
    for x in input:
        if x < mean + 3 * std and x > mean - 3 * std:
            processed_list.append(x)
    return processed_list

def iterated_execute_sql(predicted_sql, ground_truth, db_path, iterate_num, sql_dialect):
    """Execute SQL queries multiple times to measure performance"""
    diff_list = []
    predicted_res = execute_sql(predicted_sql, db_path, sql_dialect)
    ground_truth_res = execute_sql(ground_truth, db_path, sql_dialect)
    reward = 0
    time_ratio = 0
    
    if set(predicted_res) == set(ground_truth_res):
        for _ in range(iterate_num):
            predicted_time = execute_sql(predicted_sql, db_path, sql_dialect, return_time=True)
            ground_truth_time = execute_sql(ground_truth, db_path, sql_dialect, return_time=True)
            diff_list.append(ground_truth_time / predicted_time)
            
        processed_diff_list = clean_abnormal(diff_list)
        time_ratio = sum(processed_diff_list) / len(processed_diff_list)
        
    if time_ratio == 0:
        reward = 0
    elif time_ratio >= 2:
        reward = 1.25
    elif time_ratio >= 1 and time_ratio < 2:
        reward = 1
    elif time_ratio >= 0.5 and time_ratio < 1:
        reward = 0.75
    elif time_ratio >= 0.25 and time_ratio < 0.5:
        reward = 0.5
    else:
        reward = 0.25
    return reward

# ================ Common Execution Functions ================
def execute_model(predicted_sql, ground_truth, db_place, idx, meta_time_out, sql_dialect, metric_type, iterate_num=None):
    """Execute SQL queries and calculate results based on metric type"""
    start_time = time.time()
    try:
        if metric_type == "ves":
            reward = func_timeout(
                meta_time_out * iterate_num,
                iterated_execute_sql,
                args=(predicted_sql, ground_truth, db_place, iterate_num, sql_dialect),
            )
            result = {"sql_idx": idx, "reward": reward}
        else:
            score_func = calculate_ex if metric_type == "ex" else calculate_f1_score
            res = func_timeout(
                meta_time_out,
                execute_sql,
                args=(predicted_sql, ground_truth, sql_dialect, score_func),
            )
            result = {"sql_idx": idx, "res": res}
    except KeyboardInterrupt:
        logger.error("Evaluation interrupted by user")
        sys.exit(0)
    except FunctionTimedOut:
        logger.warning(f"Query {idx} timed out after {meta_time_out} seconds")
        result = {"sql_idx": idx, "res": 0} if metric_type != "ves" else {"sql_idx": idx, "reward": 0}
    except Exception as e:
        logger.error(f"Error processing query {idx}: {str(e)}")
        result = {"sql_idx": idx, "res": 0} if metric_type != "ves" else {"sql_idx": idx, "reward": 0}
    
    execution_time = time.time() - start_time
    logger.debug(f"Query {idx} executed in {execution_time:.2f} seconds")
    return result

def run_sqls_parallel(sqls, db_places, num_cpus=1, meta_time_out=30.0, sql_dialect="SQLite", metric_type="ex", iterate_num=None):
    """Run SQL queries in parallel and collect results"""
    global total_queries
    total_queries = len(sqls)
    
    logger.info(f"Starting parallel execution with {num_cpus} CPUs")
    logger.info(f"Total queries to process: {total_queries}")
    logger.info(f"Metric type: {metric_type}")
    if metric_type == "ves":
        logger.info(f"Iterations per query: {iterate_num}")
    
    start_time = time.time()
    pool = mp.Pool(processes=num_cpus)
    pbar = None
    
    try:
        # Create a progress bar
        pbar = tqdm(total=total_queries, desc="Processing queries")
        
        def update_progress(*args):
            if pbar is not None:
                pbar.update(1)
        
        def combined_callback(result):
            result_callback(result)
            update_progress()
        
        for i, sql_pair in enumerate(sqls):
            predicted_sql, ground_truth = sql_pair
            pool.apply_async(
                execute_model,
                args=(
                    predicted_sql,
                    ground_truth,
                    db_places[i],
                    i,
                    meta_time_out,
                    sql_dialect,
                    metric_type,
                    iterate_num,
                ),
                callback=combined_callback,
            )
        
        pool.close()
        pool.join()
        
        total_time = time.time() - start_time
        logger.info(f"Parallel execution completed in {total_time:.2f} seconds")
        logger.info(f"Average time per query: {total_time/total_queries:.2f} seconds")
        
    except KeyboardInterrupt:
        logger.error("Evaluation interrupted by user")
        if pool is not None:
            pool.terminate()
        raise
    except Exception as e:
        logger.error(f"Error during parallel execution: {str(e)}")
        if pool is not None:
            pool.terminate()
        raise
    finally:
        if pbar is not None:
            pbar.close()
        if pool is not None:
            pool.close()
            pool.join()

# ================ Result Processing Functions ================
def compute_ves(exec_results):
    """Calculate VES score from execution results"""
    num_queries = len(exec_results)
    total_reward = 0
    for result in exec_results:
        total_reward += math.sqrt(result["reward"]) * 100
    return total_reward / num_queries

def compute_metric_by_diff(exec_results, diff_json_path, metric_type):
    """Calculate metric scores by difficulty level"""
    num_queries = len(exec_results)
    contents = load_json(diff_json_path)
    simple_results, moderate_results, challenging_results = [], [], []

    logger.info(f"Processing {num_queries} execution results against {len(contents)} difficulty entries")
    
    # Validate that we have enough results
    if num_queries < len(contents):
        logger.warning(f"Warning: Number of execution results ({num_queries}) is less than number of difficulty entries ({len(contents)})")
        logger.warning("Some queries may be missing from the results")
    
    # Process results with error handling
    for i, content in enumerate(contents):
        try:
            if i >= len(exec_results):
                logger.warning(f"Missing result for query {i} in difficulty file")
                continue
                
            if content["difficulty"] == "simple":
                simple_results.append(exec_results[i])
            elif content["difficulty"] == "moderate":
                moderate_results.append(exec_results[i])
            elif content["difficulty"] == "challenging":
                challenging_results.append(exec_results[i])
            else:
                logger.warning(f"Unknown difficulty level '{content['difficulty']}' for query {i}")
        except Exception as e:
            logger.error(f"Error processing query {i}: {str(e)}")
            continue

    # Log the distribution of results
    logger.info(f"Results distribution - Simple: {len(simple_results)}, Moderate: {len(moderate_results)}, Challenging: {len(challenging_results)}")

    # Calculate scores with error handling
    try:
        if metric_type == "ves":
            simple_score = compute_ves(simple_results) if simple_results else 0
            moderate_score = compute_ves(moderate_results) if moderate_results else 0
            challenging_score = compute_ves(challenging_results) if challenging_results else 0
            all_score = compute_ves(exec_results)
        else:
            simple_score = sum([res["res"] for res in simple_results]) / len(simple_results) * 100 if simple_results else 0
            moderate_score = sum([res["res"] for res in moderate_results]) / len(moderate_results) * 100 if moderate_results else 0
            challenging_score = sum([res["res"] for res in challenging_results]) / len(challenging_results) * 100 if challenging_results else 0
            all_score = sum([res["res"] for res in exec_results]) / num_queries * 100
    except Exception as e:
        logger.error(f"Error calculating scores: {str(e)}")
        simple_score = moderate_score = challenging_score = all_score = 0

    count_lists = [
        len(simple_results),
        len(moderate_results),
        len(challenging_results),
        num_queries,
    ]
    return simple_score, moderate_score, challenging_score, all_score, count_lists

# ================ Main Function ================
def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--predicted_sql_path", type=str, required=False, default="./exp_result/predict_mini_dev_gpt-4_cot_SQLite.json")
    args_parser.add_argument("--ground_truth_path", type=str, required=False, default="./data/mini_dev_sqlite_gold.sql")
    args_parser.add_argument("--db_root_path", type=str, required=False, default="./data/dev_databases/")
    args_parser.add_argument("--num_cpus", type=int, default=15)
    args_parser.add_argument("--meta_time_out", type=float, default=5.0)
    args_parser.add_argument("--diff_json_path", type=str, default="./data/mini_dev_sqlite.json")
    args_parser.add_argument("--sql_dialect", type=str, default="SQLite")
    args_parser.add_argument("--output_log_path", type=str, default="./eval_result/predict_mini_dev_gpt-4_cot_SQLite.txt")
    args_parser.add_argument("--metric", type=str, choices=["ex", "f1", "ves"], default="ves")
    args_parser.add_argument("--iterate_num", type=int, default=100)
    args_parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = args_parser.parse_args()

    # Set logging level
    logger.setLevel(args.log_level)
    
    # Set up file logging
    setup_file_logging(args.output_log_path)

    logger.info("Starting evaluation process")
    logger.info(f"Configuration: {vars(args)}")

    try:
        # Clear global result list
        global exec_result
        exec_result = []

        # Package SQL queries
        logger.info("Packaging SQL queries...")
        pred_queries, db_paths = package_sqls(args.predicted_sql_path, args.db_root_path, mode='pred')
        gt_queries, db_paths_gt = package_sqls(args.ground_truth_path, args.db_root_path, mode="gt")
        query_pairs = list(zip(pred_queries, gt_queries))
        logger.info(f"Packaged {len(query_pairs)} query pairs")

        # Validate query pairs
        if not query_pairs:
            logger.error("No query pairs found to evaluate")
            return

        # Run evaluation
        logger.info("Starting query execution...")
        run_sqls_parallel(
            query_pairs,
            db_places=db_paths_gt,
            num_cpus=args.num_cpus,
            meta_time_out=args.meta_time_out,
            sql_dialect=args.sql_dialect,
            metric_type=args.metric,
            iterate_num=args.iterate_num if args.metric == "ves" else None,
        )
        
        # Sort and validate results
        exec_result = sort_results(exec_result)
        if not exec_result:
            logger.error("No execution results obtained")
            return

        # Calculate and print results
        logger.info(f"Calculating {args.metric.upper()} scores...")
        simple_score, moderate_score, challenging_score, all_score, count_lists = compute_metric_by_diff(
            exec_result, args.diff_json_path, args.metric
        )
        
        # Log detailed results
        logger.info(f"Simple queries ({count_lists[0]}): {simple_score:.2f}%")
        logger.info(f"Moderate queries ({count_lists[1]}): {moderate_score:.2f}%")
        logger.info(f"Challenging queries ({count_lists[2]}): {challenging_score:.2f}%")
        logger.info(f"Overall score ({count_lists[3]}): {all_score:.2f}%")
        
        score_lists = [simple_score, moderate_score, challenging_score, all_score]
        print_data(score_lists, count_lists, metric=args.metric.upper(), result_log_file=args.output_log_path)
        
        logger.info("=" * 100)
        logger.info(f"Finished {args.metric.upper()} evaluation for {args.sql_dialect} on Mini Dev set")
        logger.info("\n\n")
        
    except KeyboardInterrupt:
        logger.error("Evaluation interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise
    finally:
        # Ensure any remaining resources are cleaned up
        logger.info("Cleaning up resources...")

if __name__ == "__main__":
    main() 