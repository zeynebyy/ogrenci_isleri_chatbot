from src.database.sqlite_db import create_table, insert_fiches
from src.config import FICHES_FILE

create_table()

insert_fiches(FICHES_FILE)

print(f"Database oluşturuldu ve fişler ({FICHES_FILE}) eklendi.")