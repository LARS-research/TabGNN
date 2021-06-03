import os

from __init__ import data_root
from data.utils import create_datapoints_with_xargs, get_neo4j_db_driver

if __name__ == '__main__':
    db_name = 'jd_single'

    driver = get_neo4j_db_driver(db_name)
    with driver.session() as session:
        datapoint_ids = session.run('MATCH (a:Main_table) RETURN a.INDEX_ID').value()

    # Variable length match since not all applications have bureaus or previous
    # base_query = 'MATCH r = (a:Main_table)-[*0..1]-(n) where a.INDEX_ID = {} RETURN a, r, n'

    base_query = 'MATCH r = (a:Main_table)-[*0..1]-(n)  \
                  where a.INDEX_ID = {} \
                  RETURN a, r, n'

    #target_dir = os.path.join(data_root, db_name, 'preprocessed_datapoints_fz2')
    target_dir = os.path.join(data_root.replace('home','data'), db_name, 'preprocessed_datapoints_fz')
    os.makedirs(target_dir, exist_ok=False)

    n_jobs = 42

    create_datapoints_with_xargs(db_name, datapoint_ids, base_query, target_dir, n_jobs)
