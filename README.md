# BIRD-SQL Mini-Dev 

[![Data Link]](https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip)


## Overview
Here, we provide a Lite version of developtment dataset: **Mini-Dev**. This mini-dev dataset is designed to facilitate efficient and cost-effective development cycles, especially for testing and refining SQL query generation models. This dataset results from community feedback, leading to the compilation of 500 high-quality text2sql pairs derived from 11 distinct databases in a development environment. To further enhance the practicality of the BIRD system in industry settings and support the development of text-to-SQL models, we make the Mini-Dev dataset available in both **MySQL** and **PostgreSQL**.

Additionally, we introduce two new evaluation metrics for the Mini-Dev dataset: the **Reward-based Valid Efficiency Score (R-VES)** and the **Soft F1-Score**. These metrics aim to evaluate the efficiency and accuracy of text-to-SQL models, respectively. It is important to note that the both metrics, currently in their beta version, applies exclusively to the Mini-Dev dataset using baseline models.

We welcome contributions and suggestions for enhancing these metrics, particularly regarding their integration into existing leaderboards. Please do not hesitate to contact us if you are interested in these developments or have any proposals for improvements.


Below are some key statistics of the mini-dev dataset:

### Difficulty Distribution
- **Simple:** 30%
- **Moderate:** 50%
- **Challenging:** 20%

### Database Distribution
- **Debit Card Specializing:** 30 instances
- **Student Club:** 48 instances
- **Thrombosis Prediction:** 50 instances
- **European Football 2:** 51 instances
- **Formula 1:** 66 instances
- **Superhero:** 52 instances
- **Codebase Community:** 49 instances
- **Card Games:** 52 instances
- **Toxicology:** 40 instances
- **California Schools:** 30 instances
- **Financial:** 32 instances

### Keywords Statistic

- **Main Body Keywords** •SELECT •FROM •WHERE •AND •OR •NOT •IN •EXISTS •IS •NULL •IIF •CASE •CASE WHEN.
- **Join Keywords** • INNER JOIN • LEFT JOIN • ON • AS.
- **Clause Keywords** • BETWEEN • LIKE • LIMIT • ORDER BY • ASC • DESC • GROUP BY •HAVING •UNION •ALL •EXCEPT •PARTITION BY •OVER.
- **Aggregation Keywords** • AVG • COUNT • MAX • MIN • ROUND • SUM.
- **Scalar Keywords** • ABS • LENGTH • STRFTIME • JULIADAY • NOW • CAST • SUBSTR • INSTR.
- **Comparison Keywords** •= •> •< •>= •<= •!=.
- **Computing Keywords** •- •+ •* •/.

## Dataset Introduction

The dataset contains the main following resources:

- `database`: The database should be stored under the [`./data/dev_databases/`](./data/dev_databases/). In each database folder, it has two components:
  - `database_description`: the csv files are manufactured to describe database schema and its values for models to explore or references.
  - `sqlite`: The database contents in BIRD.
> [!NOTE] 
> You have to download the latest dev databases in order to construct database in the MySQL and PostgreSQL. If you use the SQLite version only, you can use the original dev databases.
- `data`: Each text-to-SQL pairs with the oracle knowledge evidence is stored as a json file, i.e., `mini_dev_sqlite.json` is stored on [`./data/mini_dev_sqlite.json`](./data/mini_dev_sqlite.json). In each json file, it has three main parts:
  - `db_id`: the names of databases
  - `question`: the questions curated by human crowdsourcing according to database descriptions, database contents.
  - `evidence`: the external knowledge evidence annotated by experts for assistance of models or SQL annotators.
  - `SQL`: SQLs annotated by crowdsource referring to database descriptions, database contents, to answer the questions accurately.
- `ground-truth SQL file`: The SQL files are stored in the `data` directory:
  - SQLite: [`./data/mini_dev_sqlite_gold.sql`](./data/mini_dev_sqlite_gold.sql)
  - MySQL: [`./data/mini_dev_mysql_gold.sql`](./data/mini_dev_mysql_gold.sql)
  - PostgreSQL: [`./data/mini_dev_postgresql_gold.sql`](./data/mini_dev_postgresql_gold.sql)
- `src`: Contains source codes for evaluation and model interaction:
  - Evaluation scripts: `evaluation_ex.py`, `evaluation_ves.py`, `evaluation_f1.py`
  - Model interaction: `gpt_request.py`, `prompt.py`
  - Utility scripts: `evaluation_utils.py`, `table_schema.py`
  - Shell scripts: `run_gpt.sh`, `run_evaluation.sh`




## Mini-Dev Dataset in MySQL and PostgreSQL


You can locate the SQL queries within the `mini_dev_mysql.json` and `mini_dev_postgresql.json` files. These queries have been transpiled from the original SQLite versions using the sqlglot package, then refined manually and with GPT-4 Turbo. After downloading the Mini-Dev dataset, each database folder will contain .sql and command.script files. Follow the instructions below to set up the database in MySQL and PostgreSQL:

### MySQL
1. Download and install the MySQL from the official website: https://dev.mysql.com/downloads/mysql/
2. Set the environment variables: 
```
export PATH=$PATH:/usr/local/mysql/bin
```
3. Start the MySQL server: 
```
sudo /usr/local/mysql/support-files/mysql.server start
```
4. Login to the MySQL server and create the database (password will be the one you set during the installation)
```bash
mysql -u root -p
CREATE DATABASE BIRD;
```
5. Construct the database by run the following command (You can find MySQL version database: `BIRD_dev.sql` in the `MINIDEV_mysql` folder):
```bash
mysql -u root -p BIRD < BIRD_dev.sql
```
6. Examples that how to run mysql query in the Python (with   pymysql) can be find in the [`examples/mysql_example.ipynb`](./examples/mysql_example.ipynb) file.

7. If you encounter the error: "this is incompatible with sql_mode=only_full_group_by", you can run the following command to disable the sql_mode:
```sql
select @@global.sql_mode;
SET GLOBAL sql_mode='{EVERYTHING SHOW IN THE ABOVE COMMAND EXCEPT ONLY_FULL_GROUP_BY}';
```

### PostgreSQL
1. Download and install the postgresql from the official website: https://www.postgresql.org/download/ 
2. Download the pgAdmin4 from the official website: https://www.pgadmin.org/download/ (Recommended to monitor the database)
3. In pgADmin4/terminal create a new database called `BIRD`
4. Construct the database by run the following command (You can find PostgreSQL version database:`BIRD_dev.sql` in the `MINIDEV_postgresql` folder):
```bash
psql -U USERNAME -d BIRD -f BIRD_dev.sql
```
5. Examples that how to run mysql query in the Python (with Psycopg) can be find in the  [`examples/postgresql_example.ipynb`](./examples/postgresql_example.ipynb) file.




## In-Context Learning (ICL):

### Environment Setup:

First, you need install openai in your python environment by:

```bash
conda create -n BIRD python=3.11.5
pip install -r requirements.txt
```

### Collect results

Use this script to run the OpenAI model on the Azure cloud. (you may need to adjust parameters and paths with your preference):

```bash
cd ./src/
sh ./run_gpt.sh
```


## Evaluation:

### Execution (EX) Evaluation:

Please post-process your collected results as the format: SQL and its `db_id`, which is splitted by `'\t----- bird -----\t'`. The examples are shown in the [`./exp_result/turbo_output/predict_mini_dev_gpt-4-turbo_cot_SQLite.json`](./exp_result/turbo_output/predict_mini_dev_gpt-4-turbo_cot_SQLite.json). Put the ground-truth sql file in the [`./data/`](./data/). And you may need to design a ChatGPT tag by your own.
The main file for ex evaluation is located at [`./src/evaluation_ex.py`](./src/evaluation_ex.py). \
Then you could evaluate the results by the following command line :

```bash
cd ./src/
sh ./run_evaluation.sh
```

### Reward-based Valid Efficiency Score (R-VES):
The main file for R-VES evaluation is located at [`./src/evaluation_ves.py`](./src/evaluation_ves.py).
R-VES and EX can be evaluated in the same shell, so you can eval your efficiency via:

```bash
cd ./src/
sh ./run_evaluation.sh
```
(For stable R-VES, you may need to enlarge `timeout` or repeat and average results. In our test evaluation, we will enlarge `timeout` to 3 s/ex; then we repeat 5 times for VES computation, only the highest results will be reported.)

In the latest version, we adjust the VES evaluation to be more stable and reliable. Instead of simply measuring the time ratio between predict and ground-truth SQLs, we now assign reward point based on the time ratio. The R-VES are calculated as follows:
<p align="center" width="100%">
<a><img src="materials/time_ratio_formula.png" style="width: 70%; min-width: 300px; display: block; margin: auto;"></a>
</p>



### Soft F1-Score Evaluation:
The main file for Soft F1-Score evaluation is located at [`./src/evaluation_f1.py`](./src/evaluation_f1.py). Soft-F1, VES and EX can be evaluated in the same shell, so you can eval your efficiency via:

```bash
cd ./src/
sh ./run_evaluation.sh
```
#### Soft F1-Score:
Alongside the update to the Mini-Dev set, we introduced a new evaluation metric—the soft F1-score. This metric is specifically designed to assess the performance of text-to-SQL models by measuring the similarity between the tables produced by predicted SQL queries and those from the ground truth. In a nutshell, the soft F1-score is a more lenient metric that reduces the impact of column order and missing values in the tables produced by predicted SQL queries.

The following demonstrate how we calculate the soft F1-score. 

Ground truth SQL resulted table:
| Row  |  | |  
|:----------:|:----------:|:----------:|
| 1 | 'Apple' | 325 | 
| 2  | 'Orange' |  | 
| 3| 'Banana' | 119 |

Predicted SQL resulted table:
| Row |  | |  
|:----------:|:----------:|:----------:|
| 1 | 325 |'Apple' |  
| 2  | 191 |'Orange' |  
| 3| |'Banana' |

The soft F1-score is calculated as follows:

|  | Matched| Pred_only | Gold_only  |
|----------|:----------:|:----------:|:----------:|
| **Row 1** | 2 | 0 | 0 | 
| **Row 2** | 1 | 1 | 0 | 
| **Row 3** | 1 | 0 | 1 | 

* tp = SUM(Matched) = 4 
* fp = SUM(Pred_only) = 1
* fn = SUM(Gold_only) = 1
* Precision = tp / (tp + fp) = 4 / 5 = 0.8
* Recall = tp / (tp + fn) = 4 / 5 = 0.8
* F1 = 2 * Precision * Recall / (Precision + Recall) = 0.8