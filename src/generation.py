#!/usr/bin/env python3
import argparse
import json
import os
from openai import OpenAI
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

from prompt import generate_combined_prompts_one

def new_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def connect_gpt(engine, prompt, max_tokens, temperature, stop, client):
    """
    Function to connect to the GPT API and get the response.
    """
    MAX_API_RETRY = 10
    last_error = None
    
    for i in range(MAX_API_RETRY):
        try:
            time.sleep(2)  # Add delay between retries
            
            if engine == "gpt-3.5-turbo-instruct":
                result = client.completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                )
                result = result.choices[0].text
            else:  # gpt-4-turbo, gpt-4, gpt-4-32k, gpt-3.5-turbo
                messages = [
                    {"role": "user", "content": prompt},
                ]
                result = client.chat.completions.create(
                    model=engine,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                )
            return result
            
        except Exception as e:
            last_error = e
            print(f"Attempt {i+1}/{MAX_API_RETRY} failed: {str(e)}")
            if i < MAX_API_RETRY - 1:  # Don't sleep on the last attempt
                time.sleep(4)
    
    # If we've exhausted all retries, raise the last error
    raise Exception(f"Failed to connect to GPT API after {MAX_API_RETRY} attempts. Last error: {str(last_error)}")


def decouple_question_schema(datasets, db_root_path):
    question_list = []
    db_path_list = []
    knowledge_list = []
    for i, data in enumerate(datasets):
        question_list.append(data["question"])
        cur_db_path = db_root_path + data["db_id"] + "/" + data["db_id"] + ".sqlite"
        db_path_list.append(cur_db_path)
        knowledge_list.append(data["evidence"])

    return question_list, db_path_list, knowledge_list


def generate_sql_file(sql_lst, output_path=None):
    """
    Function to save the SQL results to a file.
    """
    sql_lst.sort(key=lambda x: x[1])
    result = {}
    for i, (sql, _) in enumerate(sql_lst):
        result[i] = sql

    if output_path:
        directory_path = os.path.dirname(output_path)
        new_directory(directory_path)
        json.dump(result, open(output_path, "w"), indent=4)

    return result


def init_client(api_key):
    """
    Initialize the OpenAI client.
    """
    return OpenAI(api_key=api_key)


def post_process_response(response, db_path):
    sql = response if isinstance(response, str) else response.choices[0].message.content
    db_id = db_path.split("/")[-1].split(".sqlite")[0]
    sql = f"{sql}\t----- bird -----\t{db_id}"
    return sql


def worker_function(question_data):
    """
    Function to process each question, set up the client,
    generate the prompt, and collect the GPT response.
    """
    prompt, engine, client, db_path, question, i = question_data
    response = connect_gpt(engine, prompt, 512, 0, ["--", "\n\n", ";", "#"], client)
    sql = post_process_response(response, db_path)
    print(f"Processed {i}th question: {question}")
    return sql, i


def collect_response_from_gpt(
    db_path_list,
    question_list,
    api_key,
    engine,
    sql_dialect,
    num_threads=3,
    knowledge_list=None,
):
    """
    Collect responses from GPT using multiple threads.
    """
    client = init_client(api_key)

    tasks = [
        (
            generate_combined_prompts_one(
                db_path=db_path_list[i],
                question=question_list[i],
                sql_dialect=sql_dialect,
                knowledge=knowledge_list[i],
            ),
            engine,
            client,
            db_path_list[i],
            question_list[i],
            i,
        )
        for i in range(len(question_list))
    ]
    responses = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_task = {
            executor.submit(worker_function, task): task for task in tasks
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_task), total=len(tasks)
        ):
            responses.append(future.result())
    return responses


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--eval_path", type=str, default="./data/mini_dev_sqlite.json")
    args_parser.add_argument("--mode", type=str, default="mini_dev")
    args_parser.add_argument("--test_path", type=str, default="")
    args_parser.add_argument("--use_knowledge", type=str, default="True")
    args_parser.add_argument("--db_root_path", type=str, default="./data/dev_databases/")
    args_parser.add_argument("--api_key", type=str, required=False, default=os.getenv("OPENAI_API_KEY"))
    args_parser.add_argument(
        "--engine", type=str, required=False, default="gpt-4"
    )
    args_parser.add_argument("--data_output_path", type=str, default="./exp_result/")
    args_parser.add_argument("--chain_of_thought", type=str, default="True")
    args_parser.add_argument("--num_processes", type=int, default=5)
    args_parser.add_argument("--sql_dialect", type=str, default="SQLite")
    args = args_parser.parse_args()

    print(
        "Start GPT request from {} for {} evaluation; SQL dialect {} Use knowledge: {}; Use COT: {}".format(
            args.engine,
            args.mode,
            args.sql_dialect,
            args.use_knowledge,
            args.chain_of_thought,
        )
    )


    eval_data = json.load(open(args.eval_path, "r"))

    question_list, db_path_list, knowledge_list = decouple_question_schema(
        datasets=eval_data, db_root_path=args.db_root_path
    )
    assert len(question_list) == len(db_path_list) == len(knowledge_list)

    if args.use_knowledge == "True":
        responses = collect_response_from_gpt(
            db_path_list,
            question_list,
            args.api_key,
            args.engine,
            args.sql_dialect,
            args.num_processes,
            knowledge_list,
        )
    else:
        responses = collect_response_from_gpt(
            db_path_list,
            question_list,
            args.api_key,
            args.engine,
            args.sql_dialect,
            args.num_processes,
        )

    if args.chain_of_thought == "True":
        output_name = (
            args.data_output_path
            + "predict_"
            + args.mode
            + "_"
            + args.engine
            + "_cot"
            + "_"
            + args.sql_dialect
            + ".json"
        )
    else:
        output_name = (
            args.data_output_path
            + "predict_"
            + args.mode
            + "_"
            + args.engine
            + "_"
            + args.sql_dialect
            + ".json"
        )
    generate_sql_file(sql_lst=responses, output_path=output_name)

    print(
        "successfully collect results from {} for {} evaluation; SQL dialect {} Use knowledge: {}; Use COT: {}".format(
            args.engine,
            args.mode,
            args.sql_dialect,
            args.use_knowledge,
            args.chain_of_thought,
        )
    )

if __name__ == "__main__":
    main()