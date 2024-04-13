import configparser

from google.cloud import bigquery
from google.api_core.exceptions import BadRequest, NotFound, Forbidden, Conflict

config = configparser.ConfigParser()
config.read("config.ini")
GCP_PROJECT = config["Cloud Configs"]["GCP_PROJECT"]
GCP_PROJECT_LOCATION = config["Cloud Configs"]["GCP_PROJECT_LOCATION"]

class BigQueryClass():
  """Execute BigQuery queries.

  Documentation: https://cloud.google.com/python/docs/reference/bigquery/latest/index.html
  """
  def __init__(self):
    self.client = bigquery.Client(project=GCP_PROJECT)

  def run_query(self, query: str) -> str:
    print(f"Executing query...{query}")
    try:
      query_job = self.client.query(query)
    except BadRequest as e:
      print(f"ERROR: Invalid query - {str(e)}")
      return None
    except NotFound as e:
      print(f"ERROR: Resource not found - {str(e)}")
      return None
    except Forbidden as e:
      print(f"ERROR: Insufficient permissions - {str(e)}")
      return None
    except Conflict as e:
      print(f"ERROR: Concurrent modification - {str(e)}")
      return None
    except Exception as e:
      print(f"ERROR: An unexpected error occurred - {str(e)}")
      return None

    try:
      rows = query_job.result()
      result = []
      for row in rows:
        row_data = ", ".join([f"{v}" for k, v in row.items()])
        result.append(row_data)
      return "\n".join(result)
    except Exception as e:
      print(f"ERROR: Failed to retrieve query results - {str(e)}")

    return None